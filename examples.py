#from moex_utils import *
import moex_utils as moex
import pandas as pd
import matplotlib.pyplot as plt


#moex.update_all_stocks()

ticker = "LKOH"

df = moex.get_moex_stock(ticker, start='2000-01-01')
df
#df = get_moex_index('RGBI', start='2023-01-01')

dividends_df  = pd.read_csv(f"../dividends\\data\\{ticker}.csv", parse_dates= ['closing_date'])

# --- 2. Подготовка и очистка данных ---
# Удаляем строки без даты закрытия реестра или с нулевыми дивидендами
dividends_df.dropna(subset=['closing_date'], inplace=True)
dividends_df = dividends_df[dividends_df['dividend_value'] > 0]

# Сортируем дивиденды по дате
dividends_df.sort_values(by='closing_date', inplace=True)

# Создаем новый столбец для скорректированной цены и инициализируем его ценами закрытия
df['adj_close'] = df['close']

# --- 3. Расчет скорректированной цены ---
# Итерируемся по дивидендам в обратном порядке (от новых к старым)
for _, dividend in dividends_df.iloc[::-1].iterrows():
    ex_dividend_date = dividend['closing_date']
    dividend_value = dividend['dividend_value']

    # Находим позицию, куда была бы вставлена экс-дивидендная дата
    # Это позволяет найти последний торговый день, даже если сама дата - выходной
    position = df.index.searchsorted(ex_dividend_date)

    # Если див. отсечка раньше, чем наши исторические данные, пропускаем ее
    if position == 0:
        continue
    
    # Индекс предыдущего дня - это позиция минус один
    previous_day_index = position - 1
    close_before_dividend = df.iloc[previous_day_index]['close']

    # Рассчитываем коэффициент корректировки
    adjustment_factor = 1 - (dividend_value / close_before_dividend)

    # Применяем коэффициент ко всем ценам до экс-дивидендной даты
    df.loc[df.index < ex_dividend_date, 'adj_close'] *= adjustment_factor

# --- 4. Просмотр результата ---
print(df.head())
print("\n")
print(df.tail())

# --- 5. Построение графика ---
plt.style.use('seaborn-v0_8-whitegrid') # Используем приятный стиль для графика
plt.figure(figsize=(15, 7)) # Задаем размер графика

plt.plot(df.index, df['close'], label='Цена закрытия (Close)', color='blue', alpha=0.7)
plt.plot(df.index, df['adj_close'], label='Скорр. цена закрытия (Adj Close)', color='orange', linestyle='--')

# Добавляем название и подписи осей
plt.title(f"Сравнение цен Close и Adj Close для тикера {df['ticker'].iloc[0]}", fontsize=16)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Цена (RUB)', fontsize=12)

# Включаем легенду, чтобы было понятно, какая линия что означает
plt.legend(fontsize=12)

# Показываем график
plt.show()

# 6. Расчет и сравнение среднегодовой доходности (CAGR) ---

# Определяем начальные и конечные значения
start_price_close = df['close'].iloc[0]
end_price_close = df['close'].iloc[-1]

start_price_adj = df['adj_close'].iloc[0]
end_price_adj = df['adj_close'].iloc[-1]

# Рассчитываем количество лет
# Используем 365.25 для более точного учета високосных годов
start_date = df.index[0]
end_date = df.index[-1]
years = (end_date - start_date).days / 365.25

# Рассчитываем CAGR для 'close'
cagr_close = ((end_price_close / start_price_close) ** (1 / years)) - 1

# Рассчитываем CAGR для 'adj_close'
cagr_adj_close = ((end_price_adj / start_price_adj) ** (1 / years)) - 1

# --- 5. Вывод результатов ---
print("--- Сравнение среднегодовой доходности (CAGR) ---")
print(f"Период анализа: с {start_date.date()} по {end_date.date()} ({years:.2f} лет)")
print("-" * 50)
print(f"Доходность по цене закрытия (Close):       {cagr_close:.2%}")
print(f"Доходность с учетом дивидендов (Adj Close): {cagr_adj_close:.2%}")
print(f"Разница (доля дивидендов в доходности):   {cagr_adj_close - cagr_close:.2%}")
