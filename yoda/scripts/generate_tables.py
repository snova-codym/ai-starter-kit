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

# yoda/data/sec-edgar-filings/AAPL/10-Q/0000320193-23-000064/full-submission/cropped_tables/page_17_table_0.jpg

# synth_tables["Products_and_Services_Performance"] = """\\textbf{Products and Services Performance}\t\t\t\t\t\t
# \t\t\\textbf{Three Months Ended}\t\t\\textbf{Six Months Ended}\t\t
# \\hline\t\t\t\t\t\t
# ****\t\\textbf{April 1,}\t\\textbf{March 26,}\t\t\\textbf{April 1,}\t\\textbf{March 26,}\t
# ****\t\\textbf{2023}\t\\textbf{2022}\t\\textbf{Change}\t\\textbf{2023}\t\\textbf{2022}\t\\textbf{Change}
# \\hline\t\t\t\t\t\t
# Net sales by category\t\t\t\t\t\t
# *iPhone***\t$***51,334\t$***50,570\t*****2%\t$**117,109\t$**122,198\t****(4)%
# *Mac***\t*****7,168\t****10,435\t****(31)%\t***14,903\t***21,287\t***(30)%
# *iPad***\t*****6,678\t****7,646\t****(13)%\t***16,066\t***14,894\t*****8%
# *Wearables, Home and Accessories\t*****8,757\t****8,806\t*****(1)%\t***22,239\t***23,507\t***(5)%
# *Services***\t****20,907\t***19,821\t*******5%\t***41,673\t***39,337\t****6%
# \\hline\t\t\t\t\t\t
# **Total new sales**\t$***94,836\t$***97,278\t*****(3)%\t$**211,990\t$**221,223\t*****(4)%
# """

synth_tables["Products_and_Services_Performance"] = "\\textbf{Galactic Operations Performance}\t\t\t\t\t\t\n\t\t\\textbf{Four Cycles Ended}\t\t\\textbf{Eight Cycles Ended}\t\t\n\\hline\t\t\t\t\t\t\n****\t\\textbf{November 12,}\t\\textbf{October 28,}\t\t\\textbf{November 12,}\t\\textbf{October 28,}\t\n****\t\\textbf{2027}\t\\textbf{2026}\t\\textbf{Flux}\t\\textbf{2027}\t\\textbf{2026}\t\\textbf{Flux}\n\\hline\t\t\t\t\t\t\nStellar revenue by sector\t\t\t\t\t\t\n*Aurora Pods***\t$***34,219\t$***41,112\t*****17%\t$**93,791\t$**108,350\t****(13)%\n*Quasar Jets***\t*****9,421\t****7,389\t****22%\t****20,684\t****18,372\t*****12%\n*Nebula Tablets***\t*****5,237\t****6,189\t****(15)%\t***13,579\t***14,421\t***(6)%\n*Gravitational Gadgets, Hive and Nexus\t*****11,583\t****9,421\t*****23%\t***29,130\t***24,684\t*****18%\n*Chrono Services***\t****25,982\t***22,117\t*******18%\t***53,370\t***46,282\t*****15%\n\\hline\t\t\t\t\t\t\n**Total new revenue**\t$***86,442\t$***86,228\t*****0.3%\t$**210,554\t$**212,109\t*****(0.7)%"

# synth_tables["Products_and_Services_Performance"] = """\\textbf{Revenue Streams Performance}\t\t\t\t\t\t\n\t\t\\textbf{Three Months Ended}\t\t\\textbf{Six Months Ended}\t\t\n\\hline\t\t\t\t\t\t\n****\t\\textbf{January 15,}\t\\textbf{December 31,}\t\t\\textbf{January 15,}\t\\textbf{December 31,}\t\n****\t\\textbf{2024}\t\\textbf{2021}\t\\textbf{Change}\t\\textbf{2024}\t\\textbf{2021}\t\\textbf{Change}\n\\hline\t\t\t\t\t\t\nRevenue by segment\t\t\t\t\t\t\n*Galactic Explorers***\t$***82,191\t$***75,219\t*****9%\t$**183,452\t$**162,851\t***12%\n*Aurora Systems***\t*****13,429\t****11,982\t****12%\t***29,817\t***25,419\t***17%\n*Quantum Leaps***\t*****9,854\t*****12,109\t****(19)%\t***23,192\t***28,517\t****(19)%\n*Energy Harvesters\t*****12,351\t*****10,823\t*****14%\t***29,415\t***24,952\t***18%\n*Advanced Materials***\t****35,609\t***32,189\t*******11%\t***73,191\t***65,458\t***12%\n\\hline\t\t\t\t\t\t\n**Total revenue**\t$***153,434\t$***142,322\t*****7%\t$**338,067\t$**307,197\t***10%
# """

data = table_tools.convert_tsv_to_latex(synth_tables["Products_and_Services_Performance"])

path = os.path.join(DATA_DIRECTORY, "tmp")

table_tools._generate_images(folder_name=path, 
                             data=data)