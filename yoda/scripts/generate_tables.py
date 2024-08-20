import os
import sys

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.mulitmodal_utilities.table_utils import TableTools
DATA_DIRECTORY = os.path.join(kit_dir,"data")

table_tools = TableTools()

synth_tables = {}

# Template 1

synth_tables["AAPL_cash_market_secs_tsv"] = """\t\t\t\t\tCash and\tCurrent\tNon-Current
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

data = table_tools.convert_tsv_to_latex(synth_tables["AAPL_cash_market_secs_tsv"])

path = os.path.join(DATA_DIRECTORY, "tmp")

table_tools._generate_images(folder_name=path, 
                             data=data)