synth_tables = {}

# Template 1

synth_tables["AAPL_Revenue_simple_columns"] = "\\begin{tabular}{| l c c |}\n\\hline\n"

synth_tables["AAPL_Revenue_simple_tsv_formatted"] = """\tThree Months Ended\t
\tDecember 31,\tDecember 25
\t2022\t2021
iPhone(1)\t$**65,775\t$**71,628
Mac(1)\t***7,735\t***10,852
iPad(1)\t***9,396\t***7,248
Wearables, Home and Accessories\t***13,482\t***14,701
Services(3)\t***20,766\t***19,516
*Total net sales\t**117,154\t**123,945
"""

synth_tables["Costs,Gains,Securities_columns"] = "\\begin{tabular}{| l c c c c c c c |}\n\\hline\n"

synth_tables["Costs,Gains,Securities_tsv_formatted"] = """\t\t\t\t\tCash and\tCurrent\tNon-Current
\tAdjusted\tUnrealized\tUnrealized\tFair\tCash\tMarketable\tMarketable
\tCost\tGains\tLosses\tValue\tEquivalents\tSecurities\tSecurities
\\hline\t\t\t\t\t\t\t
Cash\t$**18,546\t$***-\t$***-\t$**18,546\t$**18,546\t$***-\t$***-
Level 1:\t\t\t\t\t\t\t
*Money and market funds***\t***2,929\t****-\t****-\t***2,929\t***2,929\t****-\t****-
*Mutual funds\t****274\t****-\t****(47)\t****227\t****-\t****227\t****-
\\hline\t\t\t\t\t\t\t
**Subtotal***\t***3,203\t****-\t****(47)\t***3,156\t***2,929\t****227\t****-
\\hline\t\t\t\t\t\t\t
Level 2:\t\t\t\t\t\t\t
*U.S. Treasury securities\t**25,134\t****-\t***(1,725)\t**23,409\t****338\t***5,091\t**17,980
*U.S. agency securities\t***5,823\t****-\t****(655)\t****5,168\t****-\t****240\t***4,928
*Non-U.S. government securities\t**16,948\t****2\t***(1,201)\t**15,749\t****-\t***8,806\t***6,943
*Certificates of deposit and time deposits\t**87,148\t****9\t***(7,707)\t**79,450\t****-\t***9,023\t**70,427
*Commerical paper\t****718\t****-\t****-\t****718\t****28\t****690\t****-
*Corporate debt securities\t**87,148\t****9\t***(7,707)\t**79,450\t****-\t***9,023\t**70,427
*Municipal securities\t****921\t****-\t****(35)\t****886\t****-\t****266\t****620
*Mortgage- and asset-backed securities\t**22,553\t****-\t***(2,593)\t**19,960\t****-\t****53\t**19,907
\\hline\t\t\t\t\t\t\t
**Subtotal\t*161,312\t****11\t**(13,916)\t*147,407\t***2,171\t**24,431\t*120,805
\\hline\t\t\t\t\t\t\t
***Total\t$*183,061\t$****11\t$**(13,963)\t$*169,109\t$**23,646\t$**24,658\t$*120,805
\\hline\t\t\t\t\t\t\t
"""