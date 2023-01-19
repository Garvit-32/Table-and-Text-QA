import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# model_checkpoint = "t5-base-bs-32-scale-text2text"
model_checkpoint = "t5-base-bs-32-only-num"
# model_checkpoint = "t5-base-bs-32-derivation-scale-wo-text2text"
translator = pipeline(
    "translation_en_to_en", model=model_checkpoint, device = 0, max_length = 1024
)


# model_checkpoint = "t5-base-bs-32-derivation-scale-wo-text2text"
# translator = pipeline(
#     "text2text-generation", model=model_checkpoint, device = 0, max_length = 1024
# )

# text = "What is the amount of total sales in 2019? </s> Years Ended September 30, 2019 2018 2017 Fixed Price $  1,452.4 $  1,146.2 $  1,036.9 Other 44.1 56.7 70.8 Total sales $1,496.5 $1,202.9 $1,107.7 </s> Sales by Contract Type: Substantially all of our contracts are fixed-price type contracts. Sales included in Other contract types represent cost plus and time and material type contracts. On a fixed-price type contract, we agree to perform the contractual statement of work for a predetermined sales price. On a cost-plus type contract, we are paid our allowable incurred costs plus a profit which can be fixed or variable depending on the contract’s fee arrangement up to predetermined funding levels determined by the customer. On a time-and-material type contract, we are paid on the basis of direct labor hours expended at specified fixed-price hourly rates (that include wages, overhead, allowable general and administrative expenses and profit) and materials at cost. The table below presents total net sales disaggregated by contract type (in millions):"


# text = 'What is the change in Other in 2019 from 2018? </s> Years Ended September 30, 2019 2018 2017 Fixed Price $  1,452.4 $  1,146.2 $  1,036.9 Other 44.1 56.7 70.8 Total sales $1,496.5 $1,202.9 $1,107.7 </s> Sales by Contract Type: Substantially all of our contracts are fixed-price type contracts. Sales included in Other contract types represent cost plus and time and material type contracts. On a fixed-price type contract, we agree to perform the contractual statement of work for a predetermined sales price. On a cost-plus type contract, we are paid our allowable incurred costs plus a profit which can be fixed or variable depending on the contract’s fee arrangement up to predetermined funding levels determined by the customer. On a time-and-material type contract, we are paid on the basis of direct labor hours expended at specified fixed-price hourly rates (that include wages, overhead, allowable general and administrative expenses and profit) and materials at cost. The table below presents total net sales disaggregated by contract type (in millions):'

# text = '''What is the year on year percentage change in domestic discount rate between 2018 and 2019? </s> Domestic International September 30, September 30, 2019 2018 2019 2018 Discount rate 4.00% 3.75% 1.90% 2.80% Expected return on plan assets 3.40% 3.70% Rate of compensation increase - - % - - % </s> The following table provides the weighted average actuarial assumptions used to determine net periodic benefit costfor years ended: For domestic plans, the discount rate was determined by comparison against the FTSE pension liability index for AA rated corporate instruments. The Company monitors other indices to assure that the pension obligations are fairly reported on a consistent basis. The international discount rates were determined by comparison against country specific AA corporate indices, adjusted for duration of the obligation. The periodic benefit cost and the actuarial present value of projected benefit obligations are based on actuarial assumptions that are reviewed on an annual basis. The Company revises these assumptions based on an annual evaluation of longterm trends, as well as market conditions that may have an impact on the cost of providing retirement benefits.'''

# text = 'What were the components making up current assets? </s> 2019 2018 Note £m £m Fixed assets Investments 3 1,216.0 1,212.9 1,216.0 1,212.9 Current assets Debtors 4 415.9 440.7 Cash and cash equivalents 5 – 0.2 415.9 440.9 Creditors: amounts falling due within one year 6 (411.4) (288.4) Net current assets 4.5 152.5 Net assets 1,220.5 1,365.4 Capital and reserves Called-up share capital 9 9.3 9.5 Own shares held 10 (16.5) (16.9) Capital redemption reserve 0.7 0.5 Retained earnings 1,227.0 1,372.3 Total equity 1,220.5 1,365.4 </s> Company balance sheet At 31 March 2019 The financial statements were approved by the Board of Directors on 6 June 2019 and authorised for issue.'

# text = 'What was the average Interest income for 2018 and 2019? </s> Fiscal years ended July 31, 2019 2018 Change Amount Amount ($) (%) (In thousands, except percentages) Interest income $30,182 $13,281 16,901 127 Interest expense $(17,334) $(6,442) (10,892) 169 Other income (expense), net $(1,867) $509 (2,376) (467) </s> Other Income (Expense) Interest Income Interest income represents interest earned on our cash, cash equivalents, and investments. Interest income increased by $16.9 million in fiscal year 2019. The increase in our interest income is associated with the increase in invested funds, primarily as a result of proceeds of approximately $600 million related to the common stock and convertible note offering in March 2018 and, to a lesser extent, higher yields on those invested funds. Interest Expense Interest expense includes both stated interest and the amortization of debt discount and issuance costs associated with the $400.0 million aggregate principal amount of our Convertible Senior Notes that were issued in March 2018. Accordingly, interest expense in fiscal year 2019 is higher than fiscal year 2018 as the notes were only outstanding for part of fiscal year 2018. Interest expense increased $10.9 million in fiscal year 2019, compared to the same period a year ago. Interest expense for fiscal year 2019 consists of noncash interest expense of $12.2 million related to the amortization of debt discount and issuance costs and stated interest of $5.0 million. Other Income (Expense), Net Other income (expense), net consists primarily of foreign exchange gains and losses resulting from fluctuations in foreign exchange rates on monetary asset and monetary liability balances that are denominated in currencies other than the functional currency of the entity in which they are recorded. We currently have entities with a functional currency of the Argentine Peso, Australian Dollar, Brazilian Real, British Pound, Canadian Dollar, Euro, Japanese Yen, Malaysian Ringgit, and Polish Zloty. We realized a net currency exchange loss of $1.9 million in fiscal year 2019 as compared to a net currency exchange gain of $0.5 million in fiscal year 2018 as a result of exchange rate movements on foreign currency denominated accounts against the US Dollar.'


# text = 'What is the percentage change in the final dividend from 2018 to 2019? </s> Consolidated 2019 2018 US$’000 US$’000 Final dividend for the year ended 30 June 2018 of AU 14 cents (2017: AU 12 cents) 13,327 12,534 Interim dividend for the half year ended 31 December 2018 of AU 16 cents (2017: AU 13 cents) 14,801 13,099 28,128 25,633 </s> Note 21. Equity - dividends Dividends paid during the financial year were as follows: The Directors have declared a final dividend of AU 18 cents per share for the year ended 30 June 2019. The dividend will be paid on 25 September 2019 based on a record date of 4 September 2019. This amounts to a total dividend of US$15.9 million based on the number of shares outstanding. Accounting policy for dividends Dividends are recognised when declared during the financial year and no longer at the discretion of the company.'


text = 'What was the percentage change in the amount Outstanding at 1 April in 2019 from 2018? </s> 2019 2018 Number Number Outstanding at 1 April 3,104,563 2,682,738 Options granted in the year 452,695 1,188,149 Dividend shares awarded 9,749 – Options forfeited in the year (105,213) (766,324) Options exercised in the year (483,316) – Outstanding at 31 March 2.978,478 3,104,563 Exercisable at 31 March 721,269 – </s> The number of options outstanding and exercisable as at 31 March was as follows: The weighted average market value per ordinary share for PSP options exercised in 2019 was 445.0p (2018: n/a). The PSP awards outstanding at 31 March 2019 have a weighted average remaining vesting period of 0.8 years (2018: 1.2 years) and a weighted average contractual life of 7.6 years (2018: 8.2 years).'


# inputs = f'question: {question} context: {text}'


print(translator(text, max_length = 1024))

