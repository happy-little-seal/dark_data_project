import streamlit as st
import pandas as pd
import numpy as np
import joblib # 假設您使用 joblib 儲存模型

# 載入模型
try:
    bagging_model = joblib.load('latest_rental_predictor_1214.pkl')
except FileNotFoundError:
    st.error("錯誤：模型檔案 'latest_rental_predictor_1214.pkl' 找不到。請確認檔案路徑是否正確。")
    st.stop()
except Exception as e:
    st.error(f"載入模型時發生錯誤: {e}")
    st.stop()

# 函式：處理建物類型輸入
def process_building_type(building_choice):
    """
    將使用者選擇的單一「建物」類別轉換為 One-Hot Encoding 格式 (0/1)。
    """
    # 建立一個包含所有四種建物類型的字典，預設值為 0
    building_features = {
        '建物_華廈': 0,
        '建物_住宅大樓': 0,
        '建物_公寓': 0,
        '建物_透天厝': 0
    }
    
    # 根據使用者的選擇，將對應的欄位值設為 1
    if building_choice == "華廈":
        building_features['建物_華廈'] = 1
    elif building_choice == "住宅大樓":
        building_features['建物_住宅大樓'] = 1
    elif building_choice == "公寓":
        building_features['建物_公寓'] = 1
    elif building_choice == "透天厝":
        building_features['建物_透天厝'] = 1
        
    return building_features

def process_region(region_choice):
    """
    將使用者選擇的單一「行政區」類別轉換為 One-Hot Encoding 格式 (0/1)。
    """
    # 建立一個包含所有四種行政區的字典，預設值為 0
    region_features = {
        '北屯區': 0,
        '北區': 0,
        '中區': 0,
        '西區': 0
    }
    
    # 根據使用者的選擇，將對應的欄位值設為 1
    if region_choice == "北屯區":
        region_features['北屯區'] = 1
    elif region_choice == "北區":
        region_features['北區'] = 1
    elif region_choice == "中區":
        region_features['中區'] = 1
    elif region_choice == "西區":
        region_features['西區'] = 1
        
    return region_features


## 📝 網頁配置與標題
st.set_page_config(page_title="租屋價格預測器", layout="centered")
st.title("🏠 英才周邊租屋市場預測器")
st.markdown("請輸入理想的房屋條件以預測租屋價格(月)。")
st.markdown("---")

## 💻 建立輸入表單
with st.form("rental_prediction_form"):
    st.header("1. 地點與基本資訊")
    
    # (1) 鄉鎮市區 (下拉式選單 0 或 1)
    region_choice = st.selectbox(
        "行政區",
        options=["北屯區", "北區", "中區", "西區"],
        help="此選項會自動轉換為北屯區, 北區, 中區, 西區 的 0/1 特徵。"
    )
    # (2) 車位 (下拉式選單 0 或 1)
    parking_space = st.selectbox(
        "車位 (0=無, 1=有)",
        options=[0, 1]
    )
    
    # 距離 (km) (數值輸入)
    distance = st.number_input(
        "到立夫樓的距離 (km)",
        min_value=1.0,
        #預設起始值
        value=1.5,
        step=0.1,
        help="到英才的距離。"
    )
    
    st.markdown("---")
    st.header("2. 房屋結構與狀態")

    # (3) 建物 (新的下拉式選單)
    building_type_choice = st.selectbox(
        "建物類型",
        options=["住宅大樓", "華廈", "公寓", "透天厝"],
        help="此選項會自動轉換為建物_華廈, 建物_住宅大樓, 建物_公寓, 建物_透天厝 的 0/1 特徵。"
    )
    
    # 建物總面積平方公尺 (數值輸入)
    total_area = st.number_input(
        "建物總面積 (平方公尺)",
        min_value=10.0,
        value=70.0,
        step=5.0
    )
    
    # 屋齡 (數值輸入)
    house_age = st.number_input(
        "屋齡",
        min_value=0,
        value=15,
        step=1
    )
    
    # 房廳衛數 (數值輸入)
    rooms_baths = st.number_input(
        "房廳衛數 (例如：3房2廳2衛總和為7)",
        min_value=1,
        value=5,
        step=1,
        help="房、廳、衛數量加總。"
    )
    
    st.markdown("---")
    st.header("3. 狀態與管理")

    # (4) 建物現況格局-隔間 (下拉式選單 0 或 1)
    partition = st.selectbox(
        "建物現況格局-隔間 (0=無隔間, 1=有隔間)",
        options=[0, 1]
    )

    # (4) 有無管理組織 (下拉式選單 0 或 1)
    management = st.selectbox(
        "有無管理組織 (0=無, 1=有)",
        options=[0, 1]
    )
    
    # (4) 有無附傢俱 (下拉式選單 0 或 1)
    furniture = st.selectbox(
        "有無附傢俱 (0=無, 1=有)",
        options=[0, 1]
    )
    
    # 提交按鈕
    submitted = st.form_submit_button("預測租屋價格")

## 處理表單提交
if submitted:
    
    TRAINING_FEATURES = [
        '建物總面積平方公尺', '建物現況格局-隔間', '有無管理組織', '有無附傢俱', 
        '距離 (km)', '北屯區', '北區', '中區', '西區', 
        '車位', '建物_華廈', '建物_住宅大樓', '建物_公寓', 
        '建物_透天厝', '屋齡', '房廳衛數'
    ]
    
    # 處理建物類型 One-Hot Encoding
    building_features = process_building_type(building_type_choice)
    
    # 處理行政區 One-Hot Encoding
    region_features = process_region(region_choice)

    # 建立輸入特徵 DataFrame
    input_data = {
        '建物總面積平方公尺': total_area,
        '建物現況格局-隔間': partition,
        '有無管理組織': management,
        '有無附傢俱': furniture,
        # '總額元': 0, # 這是目標，不作為輸入
        '距離 (km)': distance,
        '車位': parking_space,
        '屋齡': house_age,
        '房廳衛數': rooms_baths
    }
    
    # 加入 One-Hot Encoding 的建物特徵
    input_data.update(building_features)
    
    # 加入 One-Hot Encoding 的行政區特徵
    input_data.update(region_features)

    # 轉換成 DataFrame，確保欄位順序與訓練模型時一致
    # ⚠️ 請確保此處的欄位順序和名稱與您訓練模型時的 X.columns 完全一致
    # 這裡假設您的 X.columns 順序與輸入字典順序相似
    input_df = pd.DataFrame([input_data])
    
    try:
        final_input_df = input_df.reindex(columns=TRAINING_FEATURES, fill_value=0)
    except KeyError as e:
        st.error(f"欄位匹配錯誤：您的輸入特徵名稱與訓練特徵名稱不完全匹配。缺少：{e}")
        st.stop()
        # 顯示最終輸入的 DataFrame 結構（除錯用）
    try:
        # 進行預測
        prediction = bagging_model.predict(final_input_df)[0]
        
        st.success("✅ 預測成功！")
        st.balloons()
        
        # 輸出結果
        st.markdown(f"## 預測租屋總額 (元) 為：**NT${prediction:,.0f}**")
        
        st.markdown("---")
        st.subheader("輸入特徵總覽:")
        st.write(final_input_df)
        
    except Exception as e:
        st.error(f"模型預測時發生錯誤：請檢查輸入欄位和模型訓練時的特徵是否匹配。錯誤訊息: {e}")