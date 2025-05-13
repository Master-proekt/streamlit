import streamlit as st
import pandas as pd
import numpy as np
import psycopg2

import time
from datetime import timedelta
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt

import re
import io
import requests
import os
import os.path

from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from statsmodels.tsa.stattools import adfuller

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import base64
import pickle
import json
import base64, json, pickle


import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from causalimpact import CausalImpact

# Заголовок и интерфейс
st.set_page_config(page_title="Оценка эффекта от добавления в подписку", layout="wide")
st.title("Оценка эффекта от добавления в подписку")

# Вводные параметры
art_input = st.text_input("Перечень id книг (через запятую):")
interval = st.slider("Интервал до добавления в подписку (дней):", 30, 360, 360, step=30)
interval_after_add = st.slider("Интервал после добавления в подписку (дней):", 30, 90, 90, step=10)
dwh_login = st.secrets["DWH_LOGIN"]
dwh_password = st.secrets["DWH_PASSWORD"]

run_button = st.button("Запустить анализ") 

if run_button:
    if not art_input.strip():
        st.warning("Введите хотя бы один ID книги для запуска анализа")
    else:
        try:
            arts_lst = tuple(map(str.strip, art_input.split(",")))
            arts_lst = tuple(str(x) for x in arts_lst)

            with st.spinner("Загружаем данные и рассчитываем..."):

                date_column = 'litres_subscription_dt'
                date_column_2 = 'litres_subscription_end_dt'

                def get_data_dwh(sql, dwh_password, product='dwh'):
                    conn = psycopg2.connect(host="178.170.195.221",
                                            user=dwh_login,
                                            password=dwh_password,
                                            database="litres_dwh_prod",
                                            port=8000)
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    columns = [i[0] for i in cursor.description]
                    res = cursor.fetchall()
                    conn.close()
                    if len(res) > 0:
                        return pd.DataFrame({columns[i]: list(zip(*res))[i] for i in range(len(columns))})
                    else:
                        return pd.DataFrame({columns[i]: [] for i in range(len(columns))})


                q = f"""
                SELECT DISTINCT 
                    da.art_id,
                    da.unique_id AS hub_art_id,
                    da.art_name,
                    STRING_AGG(DISTINCT daa.author_name, ', ') AS author_names,
                    da.art_price,
                    da.art_price_crcy_code,
                    dg.nonfiction_genre,
                    dg.genre_category,
                    dat.type_name,
                    da.litres_subscr_ind,
                    da.litres_subscr_start_date,
                    da.litres_subscr_end_date,
                    da.first_time_sale,
                    STRING_AGG(DISTINCT daa2.agent_name, ', ') AS agent_name
                FROM dwh.dim_arts da
                LEFT JOIN dwh.dim_genres dg ON dg.genre_id = da.main_genre_id 
                LEFT JOIN dwh.dim_art_types dat ON dat.art_type_id = da.art_type_id 
                LEFT JOIN dwh.dim_art_authors daa ON daa.art_id = da.art_id
                LEFT JOIN dwh.dim_art_agents daa2 ON daa2.art_id = da.art_id
                WHERE da.unique_id IN {arts_lst}
                GROUP BY     
                    da.art_id,
                    da.unique_id,
                    da.art_name,
                    da.art_price,
                    da.art_price_crcy_code,
                    dg.nonfiction_genre,
                    dg.genre_category,
                    dat.type_name,
                    da.litres_subscr_ind,
                    da.litres_subscr_start_date,
                    da.litres_subscr_end_date,
                    da.first_time_sale
                """

                causalimpact_check_books = get_data_dwh(q, dwh_password)

                # Преобразование столбцов с датами
                date_columns = ['litres_subscription_dt', 'litres_subscription_end_dt']
                for col in date_columns:
                    if col in causalimpact_check_books.columns:
                        causalimpact_check_books[col] = pd.to_datetime(causalimpact_check_books[col]).dt.date

                # Преобразование столбцов с временными метками
                timestamp_columns = ['litres_subscr_start_date', 'litres_subscr_end_date', 'first_time_sale']
                for col in timestamp_columns:
                    if col in causalimpact_check_books.columns:
                        causalimpact_check_books[col] = pd.to_datetime(causalimpact_check_books[col]).dt.to_pydatetime()

                causalimpact_check_books['hub_art_id'] = causalimpact_check_books['hub_art_id'].astype(int)
                causalimpact_check_books['art_id'] = causalimpact_check_books['art_id'].astype(int)


                causalimpact_check_books['litres_subscription_dt'] = causalimpact_check_books['litres_subscr_start_date'].dt.date
                causalimpact_check_books['litres_subscription_end_dt'] = causalimpact_check_books['litres_subscr_end_date'].dt.date


                grouped_books = causalimpact_check_books.groupby(date_column)

                ppd_purchase = pd.DataFrame(columns=['sale_id', 'dt', 'art_id', 'user_id', 'hub_user_id', 'revenue_amount', 'business_model_id'])

                for date_value, books in grouped_books:
                    # Получаем список ID книг
                    book_ids = books['art_id'].tolist()

                    # Преобразуем в строку для SQL IN, учитывая одиночный элемент
                    if len(book_ids) == 1:
                        book_ids_str = f"({book_ids[0]})"
                    else:
                        book_ids_str = str(tuple(book_ids))

                    q = f"""
                    SELECT DISTINCT fsi.sale_id
                           , fsi.sale_dt AS dt
                           , fsi.art_id 
                           , fsi.user_id 
                           , du.unique_id AS hub_user_id
                           , fsi.real_payed_rub AS revenue_amount
                           , fsi.business_model_id
                    FROM dwh.fact_sale_items fsi 
                    INNER JOIN dwh.dim_users du ON fsi.user_id = du.user_id 
                    INNER JOIN dwh.dim_arts da ON da.art_id = fsi.art_id
                    WHERE 1=1 
                          AND fsi.sale_dt >= date('{date_value}') - INTERVAL '{interval} days'
                          AND fsi.sale_dt < '2025-04-26'
                          AND fsi.business_model_id = 5
                          AND fsi.real_payed_rub > 1
                          AND da.unique_id IN {arts_lst}
                    """
                    df = get_data_dwh(q, dwh_password)
                    ppd_purchase = pd.concat([ppd_purchase, df]).reset_index(drop=True)


                ppd_purchase = ppd_purchase.dropna().reset_index(drop=True)
                ppd_purchase[['sale_id', 'art_id', 'user_id', 'hub_user_id', 'business_model_id']] = ppd_purchase[
                    ['sale_id', 'art_id', 'user_id', 'hub_user_id', 'business_model_id']].astype(int)
                ppd_purchase['dt'] = pd.to_datetime(ppd_purchase['dt'], format='%Y-%m-%d')


                subs_rev = pd.DataFrame(columns=['sale_id', 'dt', 'art_id', 'user_id', 'hub_user_id', 'revenue_amount', 'business_model_id'])

                for date_value, books in grouped_books:
                    # Получаем список ID книг
                    book_ids = books['art_id'].tolist()

                    # Преобразуем в строку для SQL IN, учитывая одиночный элемент
                    if len(book_ids) == 1:
                        book_ids_str = f"({book_ids[0]})"
                    else:
                        book_ids_str = str(tuple(book_ids))

                    q = f"""
                    SELECT DISTINCT fr.sale_id
                           , fr.reading_date::date AS dt
                           , fr.art_id 
                           , fr.user_id 
                           , du.unique_id AS hub_user_id
                           , fr.revenue_amount_rub AS revenue_amount
                           , fr.business_model_id
                    FROM dwh.fact_revenue fr 
                    INNER JOIN dwh.dim_users du ON fr.user_id = du.user_id 
                    INNER JOIN dwh.dim_arts da ON da.art_id = fr.art_id
                    WHERE 1=1 
                          AND fr.reading_date::date >= date('{date_value}') - INTERVAL '{interval} days' 
                          AND date(fr.reading_date) < '2025-04-26'
                          AND fr.business_model_id = 6
                          AND fr.revenue_type_id = 10
                          AND da.unique_id IN {arts_lst}
                    """
                    df = get_data_dwh(q, dwh_password)
                    subs_rev = pd.concat([subs_rev, df]).reset_index(drop=True)


                subs_rev[['sale_id', 'art_id', 'user_id', 'hub_user_id', 'business_model_id']] = subs_rev[
                    ['sale_id', 'art_id', 'user_id', 'hub_user_id', 'business_model_id']].astype(int)
                subs_rev['revenue_amount'] = subs_rev['revenue_amount'].astype(float)
                subs_rev['dt'] = pd.to_datetime(subs_rev['dt'], format='%Y-%m-%d')


                # объединяем покупки
                rev = pd.concat([ppd_purchase, subs_rev]).reset_index(drop=True)
                rev = rev.drop_duplicates().reset_index(drop=True)


                causalimpact_check_books = causalimpact_check_books.dropna(subset=['art_id'])


                rev = rev.merge(causalimpact_check_books[['art_id', 'hub_art_id', date_column, date_column_2, 'first_time_sale']],
                                on='art_id', how='left')

                rev[date_column] = pd.to_datetime(rev[date_column])
                rev[date_column_2] = pd.to_datetime(rev[date_column_2])
                rev['first_time_sale'] = pd.to_datetime(rev['first_time_sale'])
                rev['first_time_sale'] = rev['first_time_sale'].dt.date

                # формируем датасет для расчета
                rev2 = rev.groupby(['art_id', 'hub_art_id', date_column, date_column_2, 'first_time_sale', 'dt'], dropna=False).agg(
                    {
                        'revenue_amount': 'sum',
                        'user_id': 'nunique'
                    }).reset_index()

                rev2 = rev2.rename(columns={'revenue_amount': 'sum_rev_dt', 'user_id': 'users_cnt'})

                rev2['dt'] = pd.to_datetime(rev2['dt'], format='%Y-%m-%d')
                rev2[date_column] = pd.to_datetime(rev2[date_column], format='%Y-%m-%d')
                rev2[date_column_2] = pd.to_datetime(rev2[date_column_2], format='%Y-%m-%d')

                rev2 = rev2.replace({np.NaN: None, pd.NaT: None})


                # Для каждого столбца преобразуем к datetime
                for col in date_columns:
                    rev2[col] = pd.to_datetime(rev2[col], errors='coerce')

                rev2 = rev2.drop_duplicates().reset_index(drop=True)


                #ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ ЭФФЕКТА
                def check_stationarity(series):
                    """Проверка на стационарность ряда"""
                    # Обрабатываем пропуски и бесконечности
                    series = series.replace([np.inf, -np.inf], np.nan).dropna()

                    # Проверяем, есть ли данные после удаления NaN и inf
                    if len(series) == 0 or all(series == 0):
                        return False

                    # Проверка на константность ряда
                    if series.nunique() == 1:
                        print("Ряд является константным")
                        return True  # Считаем ряд стационарным

                    # Проверяем минимальное количество данных для adfuller
                    if len(series) < 14:  # или другое подходящее минимальное число
                        print("Недостаточно данных для теста ADF.")
                        # Если хотим продолжить анализ, можно вернуть True или False.
                        # Например, вернем True, предполагая стационарность при недостатке данных:
                        return True

                    # Выполняем тест на стационарность
                    p_value = adfuller(series)[1]
                    if p_value < 0.05:
                        return True  # Стационарный
                    return False  # Нестационарный


                def regularize_series(df, custom_min_date, custom_max_date):
                    """Добавляем пропущенные даты и заполняем их нулями"""

                    full_date_range = pd.date_range(start=custom_min_date,
                                                    end=custom_max_date,
                                                    freq='D')

                    # Установим частоту для временного индекса как ежедневную
                    # df = df.asfreq('D', fill_value=0)
                    df = df.reindex(full_date_range, fill_value=0)
                    return df

                def extract_p_value(report):
                    match = re.search(r'Bayesian one-sided tail-area probability p = ([0-9.eE-]+)', report)
                    if match:
                        p_value = float(match.group(1))
                        return p_value
                    return None

                def perform_causal_impact_analysis(data, training_start, training_end, treatment_start, treatment_end, metric,
                                                   art_id=None):
                    # Фильтруем данные для тренировочного периода, используя индекс
                    df_training = data[data.index <= pd.to_datetime(training_end)]

                    # Проверяем на стационарность
                    is_stationary = check_stationarity(df_training[metric])

                    if not is_stationary:
                        print(f"Временной ряд не является стационарным")
                        # df_training = make_stationary(df_training)
                        # is_stationary = check_stationarity(df_training[metric])

                    if is_stationary:
                        print(f"Временной ряд стационарен, выполняем анализ воздействия.")
                    else:
                        print(f"Временной ряд все еще не является стационарным после преобразования.")

                    # Убедимся, что пост-период начинается после тренировочного периода
                    post_period_start = pd.to_datetime(training_end) + pd.Timedelta(days=1)

                    # Периоды для tfcausalimpact
                    pre_period = [training_start, training_end]
                    post_period = [post_period_start, treatment_end]

                    try:
                        df_for_impact = data.loc[training_start: treatment_end]
                        impact = CausalImpact(df_for_impact, pre_period, post_period,
                                              model_args={
                                              "niter": 2000,
                                              "prior_level_sd": 0.01,
                                              "standardize_data": True,
                                              "dynamic_regression": True
                                              # "nseasons":7
                                            })

                        # Построение графика
                        impact.plot()
                        report = impact.summary('report')
                        summary_data = impact.summary_data


                        # Извлечение предсказанных значений для всего периода
                        all_period_data = impact.inferences

                        # Вычисление предсказанной выручки для постпериода
                        post_period_data = all_period_data.loc[post_period_start:treatment_end]
                        predicted_rev_post = post_period_data['complete_preds_means'].sum()  # Предсказанная выручка

                        # Построение первого графика
                        plt.figure(figsize=(12, 6))
                        plt.plot(data.index, data[metric], label=f'Факт {metric}')
                        plt.plot(all_period_data.index, all_period_data['complete_preds_means'], label=f'Предсказ. {metric}')
                        plt.fill_between(all_period_data.index,
                                         all_period_data['complete_preds_lower'],
                                         all_period_data['complete_preds_upper'],
                                         color='gray', alpha=0.2, label='95% доверительный интервал')
                        plt.axvline(x=pd.to_datetime(training_end), color='red', linestyle='--', label='Дата воздействия')
                        plt.title(f'Фактическая и предсказанная {metric} до и после воздействия')
                        plt.xlabel('Дата')
                        plt.ylabel(f'{metric}')
                        plt.legend()

                        # Сохранение первого графика в буфер (blob)
                        buf1 = io.BytesIO()
                        plt.savefig(buf1, format='png')
                        buf1.seek(0)
                        graph_path1 = buf1.getvalue()
                        plt.close()

                        # Построение второго графика
                        plt.figure(figsize=(12, 6))
                        cumulative_effect = impact.inferences['post_cum_effects_means']
                        plt.fill_between(cumulative_effect.index,
                                         impact.inferences['post_cum_effects_lower'],
                                         impact.inferences['post_cum_effects_upper'],
                                         color='gray', alpha=0.2, label='95% доверительный интервал')
                        plt.plot(cumulative_effect.index, cumulative_effect, label='Кумулятивный эффект', color='blue')
                        plt.axhline(y=0, color='black', linestyle='--')
                        plt.axvline(x=pd.to_datetime(training_end), color='red', linestyle='--', label='Дата воздействия')
                        plt.title('Кумулятивный эффект воздействия')
                        plt.xlabel('Дата')
                        plt.ylabel('Кумулятивный эффект')
                        plt.legend()

                        # Сохранение второго графика в буфер (blob)
                        buf2 = io.BytesIO()
                        plt.savefig(buf2, format='png')
                        buf2.seek(0)
                        graph_path2 = buf2.getvalue()
                        plt.close()

                        return report, summary_data, predicted_rev_post, graph_path1, graph_path2


                    except Exception as e:
                        print(f"Ошибка при выполнении CausalImpact для art_id {art_id}: {e}")

                        return None, None, None, None, None

                def analyze_books(df, date_column, date_column_2, interval, observation_date=pd.Timestamp('today').normalize() - pd.Timedelta(days=1),
                              metric='sum_rev_dt', alfa=0.05):
                    results = []

                    for art_id, group in df.groupby('art_id'):
                        print(f"Начало анализа воздействия для art_id: {art_id}")

                        # Извлекаем даты для текущей книги
                        litres_subscription_dt = pd.to_datetime(group[date_column].iloc[0])
                        litres_subscription_end_dt = group[date_column_2].iloc[0]
                        if pd.notnull(litres_subscription_end_dt):
                            litres_subscription_end_dt = pd.to_datetime(litres_subscription_end_dt)
                        else:
                            litres_subscription_end_dt = None

                        # Извлекаем first_time_sale из данных
                        first_time_sale = pd.to_datetime(group['first_time_sale'].iloc[0])

                        # Преобразуем 'dt' в datetime и устанавливаем индекс
                        group['dt'] = pd.to_datetime(group['dt'])
                        data = group[['dt', metric]].copy()
                        data = data.set_index('dt')

                        data.sort_index(inplace=True)
                        print("Sorted index:", data.index.is_monotonic_increasing)

                        # Условие для training_start
                        training_start_candidate = litres_subscription_dt - pd.Timedelta(days=interval)
                        if first_time_sale >= training_start_candidate:
                            training_start = first_time_sale
                        else:
                            training_start = training_start_candidate

                        # Проверка на последовательность дат
                        if first_time_sale > litres_subscription_dt:
                            effect_status = "Нет данных до события"
                            results.append({
                                "art_id": art_id,
                                f"p_value_{metric}": None,
                                f"abs_effect_{metric}": None,
                                f"effect_status_{metric}": effect_status,
                                f"predicted_post_{metric}": 0,
                                f"avg_predicted_{metric}": None,
                                f"avg_actual_{metric}": None,
                                f"avg_{metric}_before_event": None,
                                f"graph_path1_{metric}": None,
                                f"graph_path2_{metric}": None
                            })
                            print(effect_status)
                            continue

                        # Определяем тренировочный и тестовый периоды на основе условий
                        training_end = litres_subscription_dt - pd.Timedelta(days=1)
                        treatment_start = litres_subscription_dt

                        if litres_subscription_end_dt is not None:
                            treatment_end = litres_subscription_end_dt
                            #print('3')
                        else:
                            treatment_end = observation_date
                            #print('5', training_start, training_end, treatment_start, treatment_end)

                        # Ограничиваем treatment_end текущей датой
                        treatment_end = min(treatment_end, observation_date)
                        #print('5.1', treatment_end)

                        # Вычисляем общее количество дней в каждом периоде
                        days_before = (training_end - training_start).days + 1
                        days_after = (treatment_end - treatment_start).days + 1

                        # Если количество дней продаж до события меньше, чем после, сокращаем период после события
                        if days_before < days_after:
                            treatment_end = treatment_start + pd.Timedelta(days=90)
                            #print('6', treatment_end)
                        # Ограничиваем treatment_end текущей датой
                        treatment_end = min(treatment_end, observation_date)
                        #print(f"Новый treatment_end: {treatment_end}")

                        # Проверяем, что training_end не раньше training_start
                        if training_end < training_start:
                            effect_status = "Некорректный тренировочный период"
                            results.append({
                                "art_id": art_id,
                                f"p_value_{metric}": None,
                                f"abs_effect_{metric}": None,
                                f"effect_status_{metric}": effect_status,
                                f"predicted_post_{metric}": 0,
                                f"avg_predicted_{metric}": None,
                                f"avg_actual_{metric}": None,
                                f"avg_{metric}_before_event": None,
                                f"graph_path1_{metric}": None,
                                f"graph_path2_{metric}": None
                            })
                            print(effect_status)
                            continue

                        # Выводим даты для проверки
                        print(art_id)
                        print(f"training_start: {training_start}")
                        print(f"training_end: {training_end}")
                        print(f"treatment_start: {treatment_start}")
                        print(f"treatment_end: {treatment_end}")
                        print(f"data.index.min(): {data.index.min()}")
                        print(f"data.index.max(): {data.index.max()}")


                        # Проверка на наличие продаж
                        if data[metric].sum() == 0:
                            effect_status = "Нет продаж"
                            results.append({
                                "art_id": art_id,
                                f"p_value_{metric}": None,
                                f"abs_effect_{metric}": None,
                                f"effect_status_": effect_status,
                                f"predicted_post_{metric}": 0,
                                f"avg_predicted_{metric}": None,
                                f"avg_actual_{metric}": None,
                                f"avg_{metric}_before_event": None,
                                f"graph_path1_{metric}": None,
                                f"graph_path2_{metric}": None
                            })
                            print(effect_status)
                            continue

                        # Проверка на достаточное количество данных
                        data_before_training_end = data.loc[training_start:training_end]
                        if len(data_before_training_end) < 5:
                            effect_status = "Недостаточно данных для анализа"
                            results.append({
                                "art_id": art_id,
                                f"p_value_{metric}": None,
                                f"abs_effect_{metric}": None,
                                f"effect_status_{metric}": effect_status,
                                f"predicted_post_{metric}": 0,
                                f"avg_predicted_{metric}": None,
                                f"avg_actual_{metric}": None,
                                f"avg_{metric}_before_event": None,
                                f"graph_path1_{metric}": None,
                                f"graph_path2_{metric}": None
                            })
                            print(effect_status)
                            continue

                        # Регуляризуем временной ряд, добавляем недостающие даты
                        data = regularize_series(data, training_start, treatment_end)

                        # ---- ДОБАВЛЯЕМ КОВАРИАТУ days_since_release ----
                        # Разница (в днях) между текущей датой (индексом) и first_time_sale
                        data['days_since_release'] = np.maximum((data.index - first_time_sale).days, 0)

                        data['age_log'] = np.log1p(data['days_since_release'])

                        data = data[[metric, 'age_log']]
                        # ----------------------------------------------

                        # Проверка: если после регуляризации нет данных
                        if data.empty:
                            effect_status = "Пустой временной ряд после регуляризации"
                            results.append({
                                "art_id": art_id,
                                f"p_value_{metric}": None,
                                f"abs_effect_{metric}": None,
                                f"effect_status_{metric}": effect_status,
                                f"predicted_post_{metric}": 0,
                                f"avg_predicted_{metric}": None,
                                f"avg_actual_{metric}": None,
                                f"avg_{metric}_before_event": None,
                                f"graph_path1_{metric}": None,
                                f"graph_path2_{metric}": None
                            })
                            print(effect_status)
                            continue

                        # Проверка на дату последней продажи
                        if data.index.max() <= training_end:
                            effect_status = "Нет продаж после события"
                            results.append({
                                "art_id": art_id,
                                f"p_value_{metric}": None,
                                f"abs_effect_{metric}": None,
                                f"effect_status_{metric}": effect_status,
                                f"predicted_post_{metric}": 0,
                                f"avg_predicted_{metric}": None,
                                f"avg_actual_{metric}": None,
                                f"avg_{metric}_before_event": None,
                                f"graph_path1_{metric}": None,
                                f"graph_path2_{metric}": None
                            })
                            print(effect_status)
                            continue

                        # Выполняем анализ воздействия
                        report, summary_data, predicted_rev_post, graph_path1, graph_path2 = perform_causal_impact_analysis(
                            data, training_start, training_end, treatment_start, treatment_end, metric, art_id=art_id)

                        # Если произошла ошибка
                        if report is None:
                            effect_status = "Ошибка обработки данных"
                            results.append({
                                "art_id": art_id,
                                f"p_value_{metric}": None,
                                f"abs_effect_{metric}": None,
                                f"effect_status_{metric}": effect_status,
                                f"predicted_post_{metric}": 0,
                                f"avg_predicted_{metric}": None,
                                f"avg_actual_{metric}": None,
                                f"avg_{metric}_before_event": None,
                                f"graph_path1_{metric}": None,
                                f"graph_path2_{metric}": None
                            })
                            print(effect_status)
                            continue

                        p_value = extract_p_value(report)
                        abs_effect = summary_data.loc['abs_effect', 'average']

                        #добавляем среднее дневное значение до события
                        avg_metric_before_event = data_before_training_end[metric].sum() / days_before

                        avg_actual = summary_data.loc['actual', 'average']  # Среднедневное фактическое значение
                        avg_predicted = summary_data.loc['predicted', 'average']  # Среднедневное предсказанное значение
                        avg_diff = avg_actual - avg_predicted

                        print(art_id)
                        print("Среднедневная фактическая выручка:", avg_actual)
                        print("Среднедневная предсказанная выручка (без интервенции):", avg_predicted)
                        print("Разница между фактической и предсказанной среднедневной выручкой:", avg_diff)

                        # Определение эффекта
                        if p_value is not None and p_value < alfa:
                            if abs_effect > 0:
                                effect_status = "значимо положительный эффект"
                            elif abs_effect < 0:
                                effect_status = "значимо отрицательный эффект"
                        else:
                            effect_status = "нет эффекта"

                        print(effect_status)

                        results.append({
                            "art_id": art_id,
                            f"p_value_{metric}": p_value,
                            f"abs_effect_{metric}": abs_effect,
                            f"effect_status_{metric}": effect_status,
                            f"predicted_post_{metric}": predicted_rev_post,
                            f"avg_predicted_{metric}": avg_predicted,
                            f"avg_actual_{metric}": avg_actual,
                            f"avg_{metric}_before_event": avg_metric_before_event,
                            f"graph_path1_{metric}": graph_path1,
                            f"graph_path2_{metric}": graph_path2
                        })

                    return results, pd.DataFrame(results)
                ### КОНЕЦ ФУНКЦИЙ


                batch_size = 10
                unique_art_ids = rev2['art_id'].unique()

                # Проходимся по уникальным art_id батчами
                for start_index in range(0, len(unique_art_ids), batch_size):
                    batch_art_ids = unique_art_ids[start_index:start_index + batch_size]

                    observation_date = pd.Timestamp('today').normalize() - pd.Timedelta(days=1)

                    # Фильтруем данные по текущему батчу
                    rev2_batch = rev2[rev2['art_id'].isin(batch_art_ids)]

                    try:
                        # Выполняем анализ для текущего батча
                        results1, df1 = analyze_books(rev2_batch, date_column, date_column_2, interval, alfa=0.1)
                        #print(df1.head())

                        results2, df2 = analyze_books(rev2_batch, date_column, date_column_2, interval, metric='users_cnt',
                                                      alfa=0.1)
                        #print(df2.head())

                        df = df1.merge(df2, on='art_id', how='left')
                        #print(df.head())
                        df_final = df.merge(causalimpact_check_books, on='art_id', how='left')
                        df_final['observation_date'] = observation_date
                        #print(df_final.head())

                        new_order = [
                            'art_id', 'hub_art_id', 'art_name', 'author_names', 'art_price', 'art_price_crcy_code',
                            'nonfiction_genre', 'genre_category', 'type_name', 'litres_subscr_start_date', 'litres_subscr_end_date',
                            'first_time_sale', 'agent_name', 'litres_subscription_dt', 'litres_subscription_end_dt',
                            'p_value_sum_rev_dt', 'abs_effect_sum_rev_dt', 'effect_status_sum_rev_dt', 'predicted_post_sum_rev_dt',
                            'avg_predicted_sum_rev_dt', 'avg_actual_sum_rev_dt', 'avg_sum_rev_dt_before_event',
                            'graph_path1_sum_rev_dt', 'graph_path2_sum_rev_dt',
                            'p_value_users_cnt', 'abs_effect_users_cnt', 'effect_status_users_cnt', 'predicted_post_users_cnt',
                            'avg_predicted_users_cnt', 'avg_actual_users_cnt', 'avg_users_cnt_before_event',
                            'graph_path1_users_cnt', 'graph_path2_users_cnt', 'observation_date'
                        ]

                        df_final = df_final[new_order]

                        # Преобразование столбцов
                        date_columns = ['litres_subscription_dt', 'litres_subscription_end_dt', 'observation_date']
                        df_final[date_columns] = df_final[date_columns].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

                        timestamp_columns = ['litres_subscr_start_date', 'litres_subscr_end_date', 'first_time_sale']
                        df_final[timestamp_columns] = df_final[timestamp_columns].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S',
                                                                                        errors='coerce')

                        int_columns = ['art_id', 'hub_art_id']
                        df_final[int_columns] = df_final[int_columns].astype('Int64')

                        float_columns = [
                            'art_price', 'p_value_sum_rev_dt', 'abs_effect_sum_rev_dt', 'predicted_post_sum_rev_dt',
                            'avg_predicted_sum_rev_dt', 'avg_actual_sum_rev_dt', 'avg_sum_rev_dt_before_event',
                            'p_value_users_cnt', 'abs_effect_users_cnt', 'predicted_post_users_cnt',
                            'avg_predicted_users_cnt', 'avg_actual_users_cnt', 'avg_users_cnt_before_event'
                        ]
                        df_final[float_columns] = df_final[float_columns].astype(float)
                        #print(df_final.head())

                        df_final = df_final.replace({np.NaN: None, pd.NaT: None})

                        # Добавляем пустые поля для линков, если их нет
                        if 'link1_sum_rev_dt' not in df_final.columns:
                            df_final['link1_sum_rev_dt'] = ''
                        if 'link2_sum_rev_dt' not in df_final.columns:
                            df_final['link2_sum_rev_dt'] = ''
                        if 'link1_users_cnt' not in df_final.columns:
                            df_final['link1_users_cnt'] = ''
                        if 'link2_users_cnt' not in df_final.columns:
                            df_final['link2_users_cnt'] = ''

                        # Приводим к нужному порядку колонок
                        column_order = [
                            'art_id', 'hub_art_id', 'art_name', 'author_names', 'art_price', 'art_price_crcy_code',
                            'nonfiction_genre', 'genre_category', 'type_name', 'litres_subscr_start_date', 'litres_subscr_end_date',
                            'first_time_sale', 'agent_name', 'litres_subscription_dt', 'litres_subscription_end_dt',
                            'p_value_sum_rev_dt', 'abs_effect_sum_rev_dt', 'effect_status_sum_rev_dt', 'predicted_post_sum_rev_dt',
                            'avg_predicted_sum_rev_dt', 'avg_actual_sum_rev_dt', 'avg_sum_rev_dt_before_event',
                            'graph_path1_sum_rev_dt', 'graph_path2_sum_rev_dt', 'link1_sum_rev_dt', 'link2_sum_rev_dt',
                            'p_value_users_cnt', 'abs_effect_users_cnt', 'effect_status_users_cnt', 'predicted_post_users_cnt',
                            'avg_predicted_users_cnt', 'avg_actual_users_cnt', 'avg_users_cnt_before_event',
                            'graph_path1_users_cnt', 'graph_path2_users_cnt', 'link1_users_cnt', 'link2_users_cnt',
                            'observation_date'
                        ]
                        df_final = df_final[column_order]

                        # Загружаем графики на Google Диск:

                        def get_google_services_from_token():
                            b64 = os.getenv("GOOGLE_TOKEN_B64")
                            if not b64:
                                raise ValueError("GOOGLE_TOKEN_B64 не задана в переменных окружения")

                            raw = pickle.loads(base64.b64decode(b64))
                            creds = Credentials.from_authorized_user_info(json.loads(raw.to_json()), scopes=[
                                'https://www.googleapis.com/auth/drive'
                            ])
                            return build('drive', 'v3', credentials=creds, cache_discovery=False)

                        def upload_graphs_from_df(df_final, folder_id):
                            drive_service = get_google_services_from_token()

                            for index, row in df_final.iterrows():
                                for metric in ['sum_rev_dt', 'users_cnt']:
                                    for n in [1, 2]:
                                        blob_col = f'graph_path{n}_{metric}'
                                        link_col = f'link{n}_{metric}'

                                        graph_blob = row.get(blob_col)
                                        if not graph_blob or graph_blob == "нет данных":
                                            continue

                                        if isinstance(graph_blob, str) and graph_blob.startswith("\\x"):
                                            graph_blob = bytes.fromhex(graph_blob[2:])
                                        elif isinstance(graph_blob, str):
                                            graph_blob = graph_blob.encode("latin1")

                                        bio = io.BytesIO(graph_blob)
                                        file_metadata = {
                                            'name': f'{blob_col}_{row["art_id"]}.png',
                                            'parents': [folder_id]
                                        }

                                        media = MediaIoBaseUpload(bio, mimetype='image/png')
                                        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink').execute()

                                        drive_service.permissions().create(
                                            fileId=file['id'],
                                            body={'type': 'anyone', 'role': 'reader'}
                                        ).execute()

                                        df_final.at[index, link_col] = file['webViewLink']

                            return df_final

                        # Автоматическая загрузка
                        google_folder_id = "1a2b3c4d5e6f"  # <-- Замени на реальный ID папки
                        df_final = upload_graphs_from_df(df_final, google_folder_id)

                        st.success("Анализ завершён")
                        st.dataframe(df_final)

                        file_buffer = io.BytesIO()
                        df_final.to_excel(file_buffer, index=False, engine='openpyxl')
                        file_buffer.seek(0)

                        st.download_button("Скачать результат в Excel", data=file_buffer,
                                           file_name="causalimpact_result.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    except Exception as e:
                        print(f"Ошибка в блоке кода: {e}")

        except Exception as e:
            st.error(f"Ошибка: {e}")



