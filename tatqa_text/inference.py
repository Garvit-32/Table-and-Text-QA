from transformers import pipeline
from utils import * 

# bert-finetuned-ner
# model_checkpoint = "bert-large-num-bs-16"
model_checkpoint = "bert-large-bs-16"

# model_checkpoint = "bert-base-bs-32"



# para = 'Advertising Costs: Advertising costs amounted to $278,057, $365,859, and $378,217, for the years ended September 30, 2019, 2018, and 2017, respectively, and are charged to expense when incurred. Net Income Per Share: Basic and diluted net income per share is computed by dividing net income by the weighted average number of common shares outstanding and the weighted average number of dilutive shares outstanding, respectively. There were 268,000 and 108,000 shares for the years ended September 30, 2019 and 2018, respectively, that were excluded from the above calculation as they were considered antidilutive in nature. No shares were considered antidilutive for the year ended September 30, 2017. Use of Estimates: The preparation of financial statements in conformity with accounting principles generally accepted in the United States of America requires management to make estimates and assumptions that affect the reported amounts of assets and liabilities, related revenues and expenses and disclosure about contingent assets and liabilities at the date of the financial statements. Significant estimates include the rebates related to revenue recognition, stock based compensation and the valuation of inventory, long-lived assets, finite lived intangible assets and goodwill. Actual results may differ materially from these estimates. Recently Issued Accounting Pronouncements: In February 2016, the FASB issued ASU 2016-02, Leases. There have been further amendments, including practical expedients, with the issuance of ASU 2018-01 in January 2018, ASU 2018-11 in July 2018 and ASU 2018-20 in December 2018. The amended guidance requires the recognition of lease assets and lease liabilities by lessees for those leases classified as operating leases under previous guidance. The update is effective for annual reporting periods beginning after December 15, 2018, including interim periods within those reporting periods, with early adoption permitted. The guidance will be applied on a modified retrospective basis with the earliest period presented. Based on the effective date, this guidance will apply beginning October 1, 2019. The adoption of ASU 2016-02 will have no impact to retained earnings or net income. Upon adoption of ASU 2016-02 on October 1, 2019, we anticipate recording a right-of-use asset and an offsetting lease liability of approximately $2.3 to $2.9 million. In January 2017, the FASB issued ASU 2017-04 Intangibles-Goodwill, which offers amended guidance to simplify the accounting for goodwill impairment by removing Step 2 of the goodwill impairment test. A goodwill impairment will now be measured as the amount by which a reporting unit’s carrying value exceeds its fair value, limited to the amount of goodwill allocated to that reporting unit. This guidance is to be applied on a prospective basis effective for the Company’s interim and annual periods beginning after January 1, 2020, with early adoption permitted for any impairment tests performed after January 1, 2017. The Company does not believe the adoption of this ASU will have a material impact on our financial statements.'

para = "Sales by Contract Type: Substantially all of our contracts are fixed-price type contracts. Sales included in Other contract types represent cost plus and time and material type contracts. On a fixed-price type contract, we agree to perform the contractual statement of work for a predetermined sales price. On a cost-plus type contract, we are paid our allowable incurred costs plus a profit which can be fixed or variable depending on the contract’s fee arrangement up to predetermined funding levels determined by the customer. On a time-and-material type contract, we are paid on the basis of direct labor hours expended at specified fixed-price hourly rates (that include wages, overhead, allowable general and administrative expenses and profit) and materials at cost. The table below presents total net sales disaggregated by contract type (in millions):"

# paras = para.split(' ')
# new_paras = []
# for p in paras:
#     num = to_number(p)
#     if num is not None:
#         new_paras.append(str(w))
#     else:
#         new_paras.append(p.lower())

# new_paras = ' '.join(new_paras)


text = "What are the contract types? [SEP] " + para


token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="average", device = 0
)
output = token_classifier(text)


# output =  [{'entity_group': 'ANS', 'score': 0.94676024, 'word': 'our', 'start': 411, 'end': 414}, {'entity_group': 'ANS', 'score': 0.9590609, 'word': 'allowable incurred costs plus a profit which can be fixed or variable depending on the contract', 'start': 415, 'end': 510}, {'entity_group': 'ANS', 'score': 0.9994597, 'word': '’', 'start': 510, 'end': 511}, {'entity_group': 'ANS', 'score': 0.9950118, 'word': 's fee arrangement up to', 'start': 511, 'end': 534}, {'entity_group': 'ANS', 'score': 0.9638189, 'word': 'predetermined funding levels determined by the customer', 'start': 535, 'end': 590}, {'entity_group': 'ANS', 'score': 0.99981064, 'word': '.', 'start': 590, 'end': 591}]

print(output)


prev_end = 0

text = []
tmp = ''
for idx, i in enumerate(output):
    
    start = i['start']
    end = i['end']
    word = i['word']
    
    if prev_end + 1 == start:
        tmp += ' '

    elif prev_end != start and idx != 0:
        text.append(tmp)
        tmp = '' 

    tmp += i['word']
    if idx == len(output) - 1:
        text.append(tmp)
        
    prev_end = end
    
print(text)
    
    

    



# [{'entity_group': 'ANS', 'score': 0.94676024, 'word': 'our', 'start': 411, 'end': 414}, {'entity_group': 'ANS', 'score': 0.9590609, 'word': 'allowable incurred costs plus a profit which can be fixed or variable depending on the contract', 'start': 415, 'end': 510}, {'entity_group': 'ANS', 'score': 0.9994597, 'word': '’', 'start': 510, 'end': 511}, {'entity_group': 'ANS', 'score': 0.9950118, 'word': 's fee arrangement up to', 'start': 511, 'end': 534}, {'entity_group': 'ANS', 'score': 0.9638189, 'word': 'predetermined funding levels determined by the customer', 'start': 535, 'end': 590}, {'entity_group': 'ANS', 'score': 0.99981064, 'word': '.', 'start': 590, 'end': 591}]

# 268,000