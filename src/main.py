import os

from data import *
from stock import Stock
from functions import *

# Константы
ids_path = "../resource/brazil_ids.csv"
level_VaR = '0.95'

########################################################################################################################
# 1. Собрать данные по дневным ценам активов и дневным объёмам продаж на фондовом рынке.                               #
#    Добавить данные по индексу рынка.                                                                                 #
# 2. Преобразовать данные по ценам в данные по доходностям. Вычислить оценки ожидаемых доходностей                     #
#    и стандартных отклоненй. Постройте карту активов в системе (риск, доходность).                                    #
# 5. Задайте уровень риска и оцените Value at Risk.                                                                    #
########################################################################################################################

# Провряем существует ли файл с id доступных акций.
if not os.path.isfile(ids_path):
    print("Ids didn't found")
    exit(1)

# Загружаем данные, если они не были загружены ранее.
if not os.path.isdir('../resource/data/'):
    get_data(ids_path)

# Формируем список акций из данных. Формируем логорифмические доходности, а также вычисляем оценки.
stocks = get_Stocks(level_VaR)
#'RRRP3'
#'ALPA3','CRFB3','AZUL4', 'BPAC11', 'CMIN3', 'ENGI4','HAPV3','IRBR3', 'LWSA3', 'CASH3'
selected_stocks_names = [ 'ABEV3', 'LAME3', 'ARZZ3', 'B3SA3',  'BPAN4',
          'BBSE3', 'BRML3', 'BBDC3', 'BBDC4', 'BRAP4', 'PCAR4', 'BRKM5', 'CCRO3', 'CIEL3', 'COGN3', 
          'CPLE6', 'CSAN3', 'CPFE3',  'CVCB3', 'CYRE3', 'DXCO3', 'ECOR3', 'ENBR3', 'EMBR3',
           'ENEV3', 'EGIE3', 'EQTL3', 'EZTC3', 'FLRY3', 'GGBR4', 'GOLL4',  'HYPE3',
           'ITSA4', 'ITUB4', 'JBSS3', 'RENT3',  'LREN3', 'MGLU3', 'MRFG3']

selected_stocks = []
for name_stock in selected_stocks_names:
    print(name_stock)
    find_stock = next(stock for stock in stocks if stock.key == name_stock)
    selected_stocks.append(find_stock)

print(selected_stocks)

# Сформируем списки из оценок доходностей и рисков.
Es = []
risks = []
for stock in stocks:
    Es.append(stock.E)
    risks.append(stock.risk)

# Добавляем индекс рынка с список акций, обновляем списки доходностей и рисков.
stocks.append(get_market_index('^BVSP', level_VaR))
Es.append(stocks[-1].E)
risks.append(stocks[-1].risk)

# Выведем карту доходностей.
plot_map(Es, risks)
# -------------------------------------------------------------------------------------------------------------------- #

########################################################################################################################
# 3. Рассмотрите портфель с равными долями капитала и и отметьте его на карте активов. Дайте характеристику портфелю.  #
########################################################################################################################

# Получим средние оценки для доходностей и рисков, а также расчитаем VaR использую известные данные.
sum_Es = 0.0
sum_risks = 0.0
for i in range(len(Es)):
    sum_Es += Es[i]
    sum_risks += risks[i]

average_VaR = (min(sorted(stocks, key=lambda x: x.VaR[level_VaR])
                   [int(len(stocks) * (1.0 - float(level_VaR))):],
                   key=lambda x: x.VaR[level_VaR])).VaR[level_VaR]

# Создадим актив, который будет соответствовать раным долям капитала.
stocks.append(Stock(0, "EQAL", [], []))
stocks[-1].risk = sum_risks / len(risks)
stocks[-1].E = sum_Es / len(Es)
stocks[-1].VaR[level_VaR] = average_VaR

# Обновляем списки с рисками и доходностями.
Es.append(stocks[-1].E)
risks.append(stocks[-1].risk)

# Выведем информацию о полученном активе.
print("\nБыл сфомирован портфель с раными долями капитала. Ему соответствуют оценки E = %.3f,"
      " стандартное отклонение = %.3f, VaR = %.3f" % (stocks[-1].E, stocks[-1].risk, stocks[-1].VaR[level_VaR]))

# Отметим актив на карте.
plot_map(Es, risks, 2)
# -------------------------------------------------------------------------------------------------------------------- #

########################################################################################################################
# 4. Рассмотрите индекс рынка и отметьте его на карте активов в системе коардинат (риск, доходность).                  #
########################################################################################################################

# Отметим актив на карте.
plot_map(Es, risks, 3)
# -------------------------------------------------------------------------------------------------------------------- #

########################################################################################################################
# 5. Найти наиболее предпочтительный актив по Value at Risk. Отметить на карте активов.                                #
########################################################################################################################

# Найдём лучшую акцию по VaR
best_VaR_stock = min(stocks, key=lambda x: x.VaR[level_VaR])

# Выведем информацию о активе
print("\nБыл найден лучший актив по VaR. Ему соответсвует компания %s.\n"
      "Оценки: E = %.3f, стандартное отклонение = %.3f, VaR = %.3f"
      % (best_VaR_stock.name, best_VaR_stock.E, best_VaR_stock.risk, best_VaR_stock.VaR[level_VaR]))

# Отметим актив на карте
plot_map(Es, risks, 4, best_VaR_stock)
# -------------------------------------------------------------------------------------------------------------------- #

########################################################################################################################
# 6. Выберите несколько значимых активов. Проверить можно ли считать наблюдаемы доходности (объёмы продаж) повторными  #
#    выборками из некоторого распределения? Изучить научные подходы, выполнить проверку гипотезы о случаности.         #
########################################################################################################################

# Выберем значимые акции для рынка Бразилии.
# VALE3 - одна из крупнеших горнодобывающих компаний.
# ITUB3 - крупнейший негосударственный банк Бразилии.
# GOLL4 - один из крупнейших авиаперевозчиков Бразилии.
# VIVT3 - один из крупнейших операторов сотовой связи Бразилии.
# CMIG3 - крупный поставщик электроэнергии.
selected_stocks_names = ['VALE3', 'ITUB3', 'GOLL4', 'VIVT3', 'CMIG3']

# Найдём данные для выбранных активов.
selected_stocks = []
for name_stock in selected_stocks_names:
    find_stock = next(stock for stock in stocks if stock.key == name_stock)
    selected_stocks.append(find_stock)

# Выполним проверку гипотезы о случайности с помощью критерия инверсий. Он считается наиболее мощным.
alpha = 0.05
print('\n\nВыполним проверку гипотезы о случайности с помощью критерия инверсий на уровне alpha = ' + str(alpha) + ':')
for stock in selected_stocks:
    print('\nКомпания: ' + str(stock.name) + '. E = ' + str(round(stock.E, 3)) + '; Risk = ' +
          str(round(stock.risk, 3)) + ';')
    for target in ['profit', 'volume']:
        result, p_value = inversion_test(stock, alpha, target)
        if result:
            print(f'Г-за случайности отвергается для ' + target + ' - p_value ' + str(round(p_value, 3)))
        else:
            print(f'Г-за случайности принимается для ' + target + ' - p_value ' + str(round(p_value, 3)))

print('\n\nВыполним проверку гипотезы о случайности с помощью критерия автокорреляции'
      ' на уровне alpha = ' + str(alpha) + ':')
get_auto_correlation_plot(selected_stocks)
# -------------------------------------------------------------------------------------------------------------------- #

########################################################################################################################
# 7. Используя несколько значимых активов из разных секторов рынка, в предположении, что наблюдаемые доходности        #
# (объёмы продаж) являются повторной выборкой из некоторого распределения исследовать распределение доходностей        #
# и объёмов продаж активов.                                                                                            #
########################################################################################################################

# Построим распределения доходностей для выбранных активов
plot_vs_pdf(selected_stocks)

# Выполним проверку гипотезы о нормальности распределения с помощью тестов Anderson'а и D'Agostiono.
# С помощью t - теста выпоним проверку о том, что распределение является распределением Стьюдента.
print('\nВыполним проверку гипотезы о нормальности распределения с помощью тестов Anderson\'а и D\'Agostiono.'
      ' С помощью t - теста выпоним проверку о том, что распределение является распределением Стьюдента.\n')
for stock in selected_stocks:
    print("Выпоним тесты для {}:".format(stock.name))
    test_hypothesis(stock)

# Так как гипотеза о нормальности не отвергается для актива VALE3, построим графики и выполним анализ актива
plot_profit(stocks, 'VALE3')
plot_volume(stocks, 'VALE3')
plot_prices(stocks, 'VALE3')
print("По итогам исследования новостей, связанных с компанией VALE3, можно сделать вывод:\n"
      "Резкий рост объемов продаж и увелечение цены во втором полугодии (3 квартал),"
      " связан с тем, что компания предоставила отчет, в котором их прибыль по сравнению"
      " с предыдущим годом(2016) выросла на 47%,\n такой результат был получен благодаря рекордной добычи никеля"
      " (объем добычи достиг 365.5 млр. тон).\n"
      "Кроме этого, компания снизила производство никеля на 7.3% и пересмотрела планы добычи данного ископаемого на"
      " ближайшее 5 лет для увелечения доходности от добычи")
print('\n')
# -------------------------------------------------------------------------------------------------------------------- #

########################################################################################################################
# 8. Исследовать зависимости между дохлдностями и объёмами различных активов.                                          #
########################################################################################################################

# Проведём анализ корреляций доходностей между активами из одного сектора экономики.
# ITUB3 - крупнейший негосударственный банк Бразилии.
# BBDC3 - третий по величине банк Бразилии.
print('\nПроведём исследования корреляций для различных компаний:\n')
print_corr(stocks, 'ITUB3', 'BBDC3')


########################################################################################################################
# 8. Исследовать зависимости между дохлдностями и объёмами различных активов.                                          #
########################################################################################################################

# Проведём анализ корреляций доходностей между активами из одного сектора экономики.

# VIVT3 - крупнейший оператор сотовой связи Бразилии.
# TELB4 - одна из компаний из сферы телекоммуникация.
print_corr(stocks, 'VIVT3', 'TELB4')

# Проведём анализ корреляций доходностей между активами из разных секторов экономики.
# VIVT3 - крупнейший оператор сотовой связи Бразилии.
# CMIG3 - компания, занимающаяся электроэнергетикой Бразилии.
print_corr(stocks, 'VIVT3', 'CMIG3')

# Проведём анализ корреляций доходностей между активами из разных секторов экономики.
# VALE3 - одна из крупнеших горнодобывающих компаний.
# ITUB4 - крупнейший негосударственный банк Бразилии.
print_corr(stocks, 'VALE3', 'ITUB4')

# Проведём анализ корреляций доходностей между доходностью и объёмами актива.
# VALE3 - одна из крупнеших горнодобывающих компаний.
print_corr_profit_volume(stocks, 'VALE3')

# Проведём анализ корреляций доходностей между доходностью и объёмами актива.

# ITUB3 - крупнейший негосударственный банк Бразилии.
# BBDC3 - третий по величине банк Бразилии.
print('\nПроведём исследования корреляций для различных компаний:\n')
print_corr(stocks, 'ITUB3', 'BBDC3')

# Проведём анализ корреляций доходностей между активами из одного сектора экономики.
# VIVT3 - крупнейший оператор сотовой связи Бразилии.
# TELB4 - одна из компаний из сферы телекоммуникация.
print_corr(stocks, 'VIVT3', 'TELB4')

# Проведём анализ корреляций доходностей между активами из разных секторов экономики.
# VIVT3 - крупнейший оператор сотовой связи Бразилии.
# CMIG3 - компания, занимающаяся электроэнергетикой Бразилии.
print_corr(stocks, 'VIVT3', 'CMIG3')

# Проведём анализ корреляций доходностей между активами из разных секторов экономики.
# VALE3 - одна из крупнеших горнодобывающих компаний.
# ITUB4 - крупнейший негосударственный банк Бразилии.
print_corr(stocks, 'VALE3', 'ITUB4')

# Проведём анализ корреляций доходностей между доходностью и объёмами актива.
# VALE3 - одна из крупнеших горнодобывающих компаний.
print_corr_profit_volume(stocks, 'VALE3')

# Проведём анализ корреляций доходностей между доходностью и объёмами актива.

# CMIG3 - крупный поставщик электроэнергии.
print_corr_profit_volume(stocks, 'CMIG3')
# -------------------------------------------------------------------------------------------------------------------- #

########################################################################################################################
# 9. Построим граф, где рёбра будут символизировать сильную корреляцию между доходностями активов.                     #
########################################################################################################################
print()
get_independent_set(stocks[:-3])
plt.show()

# -------------------------------------------------------------------------------------------------------------------- #

