import pandas as pd
# Read the first CSV
wap = pd.read_csv('Final\LWAP.csv')
wap=wap[wap['REGION_NAME']=='CVIS']
# Read the second CSV
hvdc = pd.read_csv('Final/HVDC_Processed.csv')
# Check for duplicate column names and drop/rename if needed
hvdc = hvdc.drop(columns=['FLOW_MIN','FLOW_LUZ'])  # Drop duplicate RUN_TIME if not needed
rdr = pd.read_csv('Final/ReserveGWAP_DR.csv')
rfr = pd.read_csv('Final/ReserveGWAP_FR.csv')
rrd = pd.read_csv('Final/ReserveGWAP_RD.csv')
rru = pd.read_csv('Final/ReserveGWAP_RU.csv')
rdr = rdr[rdr['REGION_NAME'] == 'CVIS']
rfr = rfr[rfr['REGION_NAME'] == 'CVIS'] 
rrd = rrd[rrd['REGION_NAME'] == 'CVIS']
rru = rru[rru['REGION_NAME'] == 'CVIS']

rdr = rdr.drop(columns=['REGION_NAME'])
rfr = rfr.drop(columns=['REGION_NAME'])
rrd = rrd.drop(columns=['REGION_NAME']) 
rru = rru.drop(columns=['REGION_NAME'])
# Merge the two DataFrames horizontally (along columns)



merged_df = pd.merge(wap, hvdc, on='RUN_TIME', how='inner')
merged_df = merged_df.merge(rdr, on='RUN_TIME', how='inner')
merged_df = merged_df.merge(rfr, on='RUN_TIME', how='inner')
merged_df = merged_df.merge(rrd, on='RUN_TIME', how='inner')
merged_df = merged_df.merge(rru, on='RUN_TIME', how='inner')
print(merged_df)
merged_df.to_csv('input_lwap_lvis.csv', index=False)
