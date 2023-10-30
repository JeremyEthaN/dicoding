import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

def create_sum_order_items_df(df):
    grouped_temp1_df = all_df.groupby(["product_id","product_category_name_english", "order_id"])
    max_order_items_df = grouped_temp1_df['order_item_id'].max().reset_index()
    sum_order_items_df = max_order_items_df.groupby(by=["product_id", "product_category_name_english"]).order_item_id.sum().sort_values(ascending=False).reset_index()
    return sum_order_items_df

def create_sum_order_items_type_df(df):
    grouped_temp2_df = all_df.groupby(["product_category_name_english", "order_id"])
    max_order_items_type_df = grouped_temp2_df["order_item_id"].max().reset_index()
    sum_order_items_type_df = max_order_items_type_df.groupby("product_category_name_english")["order_item_id"].sum().sort_values(ascending=False).reset_index()
    return sum_order_items_type_df

def create_cbystate_df(df):
    cbystate_df = df.groupby(by="customer_state").customer_id.nunique().reset_index()
    cbystate_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    return cbystate_df

def create_cbycity_df(df):
    cbycity_df = df.groupby(by="customer_city_normalized").customer_id.nunique().reset_index()
    cbycity_df.rename(columns={
        "customer_id": "customer_count",
        "customer_city_normalized": "customer_city"
    }, inplace=True)
    return cbycity_df

def create_sbystate_df(df):
    sbystate_df = df.groupby(by="seller_state").seller_id.nunique().reset_index()
    sbystate_df.rename(columns={
        "seller_id": "seller_count"
    }, inplace=True)
    return sbystate_df

def create_sbycity_df(df):
    sbycity_df = df.groupby(by="seller_city_normalized").seller_id.nunique().reset_index()
    sbycity_df.rename(columns={
        "seller_id": "seller_count",
        "seller_city_normalized": "seller_city"
    }, inplace=True)
    return sbycity_df

def create_rfm_df(df):
    rfm_df = all_df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max", # mengambil tanggal order terakhir
        "order_id": "nunique", # menghitung jumlah order
        "payment_value": "sum" # menghitung jumlah revenue yang dihasilkan
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]

    # menghitung kapan terakhir pelanggan melakukan transaksi (hari)
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = all_df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)

    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    return rfm_df

all_df = pd.read_csv("data_all.csv")
datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)
 
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()
 
with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("https://raw.githubusercontent.com/JeremyEthaN/dicoding_dataset/d09da8bf643e02022cf5ae8231598032fd43a700/Dicoding%20Final%20Project/idCJV4h5Ot.png")
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    st.caption('data from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce')

main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_df["order_purchase_timestamp"] <= str(end_date))]
sum_order_items_df = create_sum_order_items_df(main_df)
sum_order_items_type_df = create_sum_order_items_type_df(main_df)
cbystate_df = create_cbystate_df(main_df)
cbycity_df = create_cbycity_df(main_df)
sbystate_df = create_sbystate_df(main_df)
sbycity_df = create_sbycity_df(main_df)
rfm_df = create_rfm_df(main_df)

st.header('Proyek Analisis Data: E-Commerce Public Dataset Dashboard :sparkles:')

with st.expander("Business Questions"):
    st.write(
        "1. Produk apa yang paling banyak terjual dan produk JENIS apa yang paling banyak dan sedikit terjual?\n"
        "2. Bagaimana Demografi khususnya lokasi tempat tinggal Pelanggan dan Penjual yang Kita Miliki?\n\n"
        "(Tambahan disertakan juga hasil visualisasi RFM)"
    )

st.subheader("Best Performing Product")

plt.figure(figsize=(10, 5))
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x="order_item_id",
    y="product_id",
    data=sum_order_items_df.head(5),
    palette=colors
)
plt.title("Best Performing Product Id by Sales", loc="center", fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

st.pyplot(plt)

st.subheader("Best & Worst Performing Product Type")
 
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

sns.barplot(x="order_item_id", y="product_category_name_english", data=sum_order_items_type_df.head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("Best Performing Product Type", loc="center", fontsize=15)
ax[0].tick_params(axis ='y', labelsize=12)

sns.barplot(x="order_item_id", y="product_category_name_english", data=sum_order_items_type_df.sort_values(by="order_item_id", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product Type", loc="center", fontsize=15)
ax[1].tick_params(axis='y', labelsize=12)
 
st.pyplot(fig)

st.subheader("Customer Geographics")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

colors_ = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3",
           "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x="customer_count",
    y="customer_state",
    data=cbystate_df.sort_values(by="customer_count", ascending=False).head(15),
    palette=colors_,
    ax=ax[0]
)
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("Number of Customer by States", loc="center", fontsize=15)
ax[0].tick_params(axis ='y', labelsize=12)

colors_ = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3",
           "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x="customer_count",
    y="customer_city",
    data=cbycity_df.sort_values(by="customer_count", ascending=False).head(15),
    palette=colors_,
    ax=ax[1]
)
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Number of Customer by Cities", loc="center", fontsize=15)
ax[1].tick_params(axis='y', labelsize=12)
 
st.pyplot(fig)

st.subheader("Seller Geographics")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

colors_ = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3",
           "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x="seller_count",
    y="seller_state",
    data=sbystate_df.sort_values(by="seller_count", ascending=False).head(15),
    palette=colors_,
    ax=ax[0]
)
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("Number of Seller by States", loc="center", fontsize=15)
ax[0].tick_params(axis ='y', labelsize=12)

sbycity_df = all_df.groupby(by="seller_city_normalized").seller_id.nunique().reset_index()
sbycity_df.rename(columns={
    "seller_id": "seller_count",
    "seller_city_normalized": "seller_city"
}, inplace=True)

colors_ = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3",
           "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x="seller_count",
    y="seller_city",
    data=sbycity_df.sort_values(by="seller_count", ascending=False).head(15),
    palette=colors_,
    ax=ax[1]
)
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("umber of Seller by States", loc="center", fontsize=15)
ax[1].tick_params(axis='y', labelsize=12)

st.pyplot(fig)

st.subheader("Best Customer Based on RFM Parameters")
 
col1, col2, col3 = st.columns(3)
 
with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)
 
with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)
 
with col3:
    avg_frequency = format_currency(rfm_df.monetary.mean(), "R$ ", locale='es_CO') 
    st.metric("Average Monetary", value=avg_frequency)
 
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

colors = ["#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4"]

sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(12), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("By Recency (days)", loc="center", fontsize=18)
ax[0].tick_params(axis ='x', labelsize=15)
plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)

sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_title("By Frequency", loc="center", fontsize=18)
ax[1].tick_params(axis='x', labelsize=15)
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)

sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel(None)
ax[2].set_title("By Monetary", loc="center", fontsize=18)
ax[2].tick_params(axis='x', labelsize=15)
plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=45)
 
st.pyplot(fig)
 
st.caption('by Jeremy Ethan Novriawan 2023')
st.caption('dicoding id: jeremyethann')