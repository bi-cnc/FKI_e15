




import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import streamlit as st
from PIL import Image
import base64
import io
import numpy as np
import re

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


st.set_page_config(
    page_title="Fondy kvalifikovaných investorů",
    page_icon="⭐",
    layout="wide"
)


# Vložení vycentrovaného obrázku

logo_image = Image.open("E15_logo_2023.png")  # Nahraďte "E15_logo_2023.png" názvem souboru vašeho loga

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(logo_image,width=600)


st.title("Fondy kvalifikovaných investorů")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("FKI_streamlit (akcie, multi asset, dluhopisy apod.) - List 1.csv")
    return df

df = load_data()

for col in df.select_dtypes(include=["object"]).columns:
    if is_categorical_dtype(df[col]):
        df[col] = df[col].str.strip() 
    
# Convert image to Base64
def image_to_base64(img_path, output_size=(441, 100)):
    # Open an image file
    with Image.open(img_path) as img:
        # Resize image
        img = img.resize(output_size)
        # Convert image to PNG and then to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    

# Apply conversion function to the column with image paths
df["Poskytovatel"] = df["Poskytovatel"].apply(image_to_base64)


# Nahraďte NaN hodnoty "Neuvedeno"
df["Vstupní poplatek"].fillna("- - -", inplace=True)
df["Manažerský poplatek/výkonnostní odměna"].fillna("- - -", inplace=True)
df["Výstupní poplatek"].fillna("- - -", inplace=True)
df["Lhůta pro zpětný odkup"].fillna("- - -", inplace=True)


def convert_yield_to_float(yield_value):
    if yield_value == "- - -":
        return -1
    if isinstance(yield_value, str):
        # Pokud obsahuje rozsah, vytvoříme kombinovanou hodnotu
        if '-' in yield_value:
            first_val, second_val = map(lambda x: float(x.replace('%', '').strip()), yield_value.split('-'))
            # Vracíme kombinovanou hodnotu
            return first_val + second_val * 0.01
        # Odeberte procenta a převeďte na float
        yield_value = yield_value.replace('%', '').replace(',', '.').strip()
        # Pokud obsahuje '+', přidáme malou hodnotu pro řazení
        if '+' in yield_value:
            yield_value = yield_value.replace('+', '').strip()
            return float(yield_value) + 0.001  # přidáme 0.001 pro řazení
        else:
            return float(yield_value)
    return None


def extract_number_from_string(s):
    numbers = re.findall(r"(\d+)", s)
    if numbers:
        return int(numbers[0])
    return 0



def convert_fee_to_float_simple(fee_value):
    if isinstance(fee_value, str):
        # Ořízne řetězec na základě první závorky (pokud existuje)
        fee_value = fee_value.split('(')[0].strip()

        # Zkusíme extrahovat čísla z řetězce
        numbers = re.findall(r"(\d+\.?\d*)", fee_value)
        if not numbers:  # pokud nejsou žádná čísla, vrátíme -1 (nebo jinou náhradní hodnotu)
            return -1

        if '%' in fee_value:
            # Pokud obsahuje více částí oddělených čárkami, vezmeme první část
            fee_value = fee_value.split(',')[0].strip()
            
            # Pokud obsahuje rozsah, vytvoříme kombinovanou hodnotu
            if '-' in fee_value:
                fee_parts = fee_value.split('-')
                # Vezmeme první číslo z rozsahu
                return float(fee_parts[0].replace('%', '').strip())
            
            # Extrakce čísla ze stringu
            fee_value = numbers[0]
            return float(fee_value)
    return -1  # Pokud nedostaneme žádnou platnou hodnotu, vrátíme -1 (nebo jinou náhradní hodnotu)




fee_columns = ["Vstupní poplatek", "Manažerský poplatek/výkonnostní odměna", "Výstupní poplatek"]



def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify2 = st.checkbox("Přidat filtrování", key="checkbox2")

    if not modify2:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:

        columns_to_exclude = ["Název fondu","Výnos 2022", "Výnos 2021","Výnos 2020", "Výnos od založení", "NAV (v mld. Kč)"]
        available_columns = [col for col in df.columns if col not in columns_to_exclude]
        to_filter_columns = st.multiselect("Filtrovat přehled podle:", available_columns,placeholder="Vybrat finanční ukazatel")
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))

            # Pro poplatky - použijeme specifické řazení
            if column in fee_columns:
                sorted_fee_values = sorted(df[column].dropna().unique(), key=convert_fee_to_float_simple)
                user_fee_input = right.multiselect(
                    column,
                    sorted_fee_values,
                    default=list(sorted_fee_values)
                )
                df = df[df[column].isin(user_fee_input)]
                continue  # pokračujte dalším sloupcem

            # Add filtering for the "Typ fondu" column
            if column == "Typ fondu":    
                unique_typy_fondu = df["Typ fondu"].dropna().unique()
                user_typ_fondu_input = right.multiselect(
                "Typ fondu",
                unique_typy_fondu,
                default=list(unique_typy_fondu)
                )
                df = df[df["Typ fondu"].isin(user_typ_fondu_input)]
            
            if column == "Lhůta pro zpětný odkup":
                sorted_lhuty_odkup = sorted(df["Lhůta pro zpětný odkup"].dropna().unique(), key=extract_number_from_string)
                user_lhuty_odkup_input = right.multiselect(
                "Lhůta pro zpětný odkup",
                sorted_lhuty_odkup,
                default=list(sorted_lhuty_odkup)
                )
                df = df[df["Lhůta pro zpětný odkup"].isin(user_lhuty_odkup_input)]

            # When creating the filter UI for this column:          
            if column == "Rok vzniku fondu":
                unique_years = [val for val in df[column].dropna().unique() if val != "- - -"]
                min_year = min(unique_years)
                max_year = max(unique_years)
                user_year_input = right.slider(
                column,
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
                )
                df = df[df[column].between(*user_year_input)]
                continue  # pokračujte dalším sloupcem
            # Pro Min. investice
            if column == "Min. investice":
                unique_values = [val for val in df[column].dropna().unique() if val != "1 mil. Kč nebo 125 tis. euro"]
                user_cat_input = right.multiselect(
                    column,
                    unique_values,
                    default=list(unique_values)
                )
                if "1 mil. Kč" in user_cat_input:
                    user_cat_input.append("1 mil. Kč nebo 125 tis. euro")
                df = df[df[column].isin(user_cat_input)]
                continue  # pokračujte dalším sloupcem

            if df[column].apply(lambda x: not pd.api.types.is_number(x)).any():
                unique_values = df[column].dropna().unique()

            elif is_numeric_dtype(df[column]):
                _min = df[column].min()
                _max = df[column].max()
                if pd.notna(_min) and pd.notna(_max):
                    _min = float(_min)
                    _max = float(_max)
    
    # Pokud jsou hodnoty min a max stejné, nevytvoříme posuvník a vrátíme dataframe filtrovaný na základě této hodnoty
                    if _min == _max:
                        df = df[df[column] == _min]
                    else:
                        step = (_max - _min) / 100
                        if step == 0:
                            step = 0.01
                        user_num_input = right.slider(
                        column,
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                        )
                        df = df[df[column].between(*user_num_input)]

            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    column,
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
 
    return df




df.rename(columns={"Výnos 2022 (v %)":"Výnos 2022 ","Výnos 2021 (v %)":"Výnos 2021 ", "Výnos 2020 (v %)":"Výnos 2020 ","Výnos od založení (% p.a.)":"Výnos od založení ","NAV (v mld. Kč)":"NAV "},inplace=True)


df.info()

def get_emoji(value):
    if isinstance(value, (int, float)) and not np.isnan(value):
        if value >= 10:
            return "🔹"
        elif value >= 5:
            return "🔸"
        elif value < 5:
            return "💢"
    return "▫"

def finance_icon(value):
    if value == "Finanční trhy":
        return "💲"
    elif value == "Akcie":
        return "📈"
    elif value == "Development":
        return "🏗️"
    elif value == "Dluhopisy":
        return "📋"
    elif value == "Pohledávky":
        return "🔍"
    elif value == "Zemědělská půda":
        return "🌱"
    elif value == "Private Equity":
        return "💡"
    elif value == "Zelená energie":
        return "♻️"
    elif value == "Úvěry":
        return "💰"
    elif value == "Různé":
        return "📝"    
    elif value == "Multi-Asset":
        return "🔄"




    else:
        return ""  # Return the original value for unmatched cases


# Apply the function to create a new column with icons and fund type names
df['Typ fondu'] = df['Typ fondu'].apply(lambda x: f"{finance_icon(x)} {x}")



# Vytvořte nový sloupec kombinující emoji a hodnotu 'Výnos 2022'
df['Výnos 2022'] = df['Výnos 2022 '].apply(lambda x: f"{get_emoji(x)} {x:.2f} %" if not np.isnan(x) else "▫️ - - -")
df['Výnos 2021'] = df['Výnos 2021 '].apply(lambda x: f"{get_emoji(x)} {x:.2f} %" if not np.isnan(x) else "▫️ - - -")
df['Výnos 2020'] = df['Výnos 2020 '].apply(lambda x: f"{get_emoji(x)} {x:.2f} %" if not np.isnan(x) else "▫️ - - -")
df['Výnos od založení'] = df['Výnos od založení '].apply(lambda x: f"{get_emoji(x)} {x:.2f} % p.a." if not np.isnan(x) else "▫️ - - -")

df["NAV (v mld. Kč)"] = df["NAV "].apply(lambda x: "- - -" if pd.isna(x) else f"{x:.2f}")


# Configure the image column
image_column = st.column_config.ImageColumn(label="Poskytovatel", width="medium")
rok_vzniku_fondu_column = st.column_config.NumberColumn(format="%d")
min_invest_column = st.column_config.TextColumn(help="📍**Minimální nutná částka pro vstup do fondu.** Klíčové zejména u FKI, kde je většinou 1 mil. Kč při splnění testu vhodnosti, ale někdy i 2 a více milionů.")
poplatky_column = st.column_config.TextColumn(help="📍**Často přehlížené, ale pro finální výnos zásadní jsou poplatky.** Je třeba znát podmínky pro výstupní poplatky v různých časových horizontech – zejména ty může investor ovlivnit.")


vynosNAV_column = st.column_config.NumberColumn(label="NAV (v mld. Kč) 💬",help="📍**NAV (AUM): Hodnota majetku fondu ukazuje na robustnost a vloženou důvěru investorů.**")


nazev_column = st.column_config.TextColumn(label="Název fondu", width="medium")

df.set_index('Poskytovatel', inplace=True)


filtered_df = filter_dataframe(df)
filtered_df.sort_values("Typ fondu",ascending=False,inplace=True)


# Seznam sloupců, které chcete přesunout na začátek
cols_to_move = ["Typ fondu","Název fondu",'Výnos 2022','Výnos 2021','Výnos 2020',"Výnos od založení","Rok vzniku fondu","Min. investice","Vstupní poplatek","Manažerský poplatek/výkonnostní odměna","Výstupní poplatek",
                "NAV (v mld. Kč)"]

# Získání seznamu všech sloupců v DataFrame a odstranění sloupců, které chcete přesunout na začátek
remaining_cols = [col for col in df.columns if col not in cols_to_move]

# Kombinování obou seznamů k vytvoření nového pořadí sloupců
new_order = cols_to_move + remaining_cols

# Přeuspořádání sloupců v DataFrame
filtered_df = filtered_df[new_order]


if not filtered_df.empty:
    st.dataframe(filtered_df.drop(columns=["Výnos 2022 ","Výnos 2021 ",'Výnos 2020 ',"Výnos od založení ","NAV "]), hide_index=True, 
                 column_config={
                     "Poskytovatel": image_column,
                     "Název fondu": nazev_column,
                     "NAV (v mld. Kč)": vynosNAV_column,
                     "Min. investice": min_invest_column,
                     "Vstupní poplatek": poplatky_column,
                     "Manažerský poplatek/výkonnostní odměna": poplatky_column,
                     "Výstupní poplatek": poplatky_column,
                     "Rok vzniku fondu": rok_vzniku_fondu_column
                 }, height=1513)
else:
    st.warning("Žádná data neodpovídají zvoleným filtrům.")


