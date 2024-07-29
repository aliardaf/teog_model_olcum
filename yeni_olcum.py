import pandas as pd
from mlx_lm import load, generate

dataframe = pd.read_csv("teog_2013_text.csv")

model = "mlx-community/SmolLM-135M-Instruct-4bit"

# Modelin testi çözmesi ve cevapların kayıdı
for i in range(len(dataframe)):
    model, tokenizer = load(model)
    response = generate(model, tokenizer, prompt=prompt_0, verbose=True)
    dataframe.at[i, "verilencevap"] = response

# Doğru ve yanlış cevapların sayı kayıdı
dogru_cevap = 0

for i in range(len(dataframe)):
    if dataframe.iloc[i]["dogrucevap"] == dataframe.iloc[i]["verilencevap"]:
        dogru_cevap += 1
        dataframe.at[i, "dogruyanlis"] = 1
    else:
        dataframe.at[i, "dogruyanlis"] = 0

# Öğrencinin başarı puanı hesaplanırken kullanılacak olan ağırlıklar
obp_6 = 80
obp_7 = 80
obp_8 = 80

agirliklar = {
    1: 4,
    2: 4,
    3: 2,
    4: 4,
    5: 2,
    6: 2,
}

# Sınav Puanının Hesaplanması 
def ders_basari_hesapla(dataframe, sinav):
    ders = dataframe[dataframe['sinav'] == sinav]
    dogru_sayisi = ders['dogruyanlis'].sum()
    return (dogru_sayisi / len(ders) * 100) * agirliklar[sinav]

def ders_basari_hesapla(dataframe, sinav):
    ders = dataframe[dataframe['sinav'] == sinav]
    dogru_sayisi = ders['dogruyanlis'].sum()
    return (dogru_sayisi / len(ders) * 100) * agirliklar[sinav]

sınav_puanı = ders_basari_hesapla(dataframe, 1) + ders_basari_hesapla(dataframe, 2) + ders_basari_hesapla(dataframe, 3) + ders_basari_hesapla(dataframe, 4) + ders_basari_hesapla(dataframe, 5) + ders_basari_hesapla(dataframe, 6)

yep = (obp_6 + obp_7 + obp_8 + sınav_puanı/100) / 2 * 3.173747
    # yep = teog puanı

sonuclar = pd.read_csv("sonuclar.csv")

sonuclar.loc[len(sonuclar)] = {"model": model, "teog 2013 puanı": yep}

sonuclar.to_csv("sonuclar.csv", index=False)

