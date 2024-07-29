import os

import pandas as pd
from mlx_lm import load, generate

dataframe = pd.read_csv("./teog_2013_text.csv")

model = "mlx-community/SmolLM-135M-Instruct-4bit"

# Modelin testi çözmesi ve cevapların kayıdı

for i in range(len(dataframe)):
    model, tokenizer = load("mlx-community/SmolLM-135M-Instruct-4bit")
    response = generate(model, tokenizer, prompt=dataframe.loc[i]['soru'] + "\n A: " + dataframe.loc[i]['cevapa'] + "\n B: " + dataframe.loc[i]['cevapb'] + "\n C: " + dataframe.loc[i]['cevapc'] + "\n D: " + dataframe.loc[i]['cevapd'] + "\n Sadece doğru şıkkın harfini söyle(A, B, C veya D), bu dört harften başka bir şey söyleme.", verbose=True)
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

turkce_basari = ders_basari_hesapla(dataframe, 1)
print(turkce_basari)
matematik_basari = ders_basari_hesapla(dataframe, 2)
print(matematik_basari)
inkilap_basari = ders_basari_hesapla(dataframe, 3)
print(inkilap_basari)
fen_basari = ders_basari_hesapla(dataframe, 4)
print(fen_basari)
ingilizce_basari = ders_basari_hesapla(dataframe, 5)
print(ingilizce_basari)
din_basari = ders_basari_hesapla(dataframe, 6)
print(din_basari)


sınav_puanı = turkce_basari + matematik_basari + inkilap_basari + fen_basari + ingilizce_basari + din_basari

yep = (obp_6 + obp_7 + obp_8 + sınav_puanı/100) / 2 * 3.173747
    # yep = teog puanı

print("yep")

sonuclar = pd.read_csv("./sonuclar.csv")

print("sonuclar okundu")

sonuclar.loc[len(sonuclar)] = {"model": "-community/SmolLM-135M-Instruct-4bit", "teog 2013 puanı": yep}

print("sonuclar yazılacak")

sonuclar.to_csv("sonuclar.csv", index=False)

print("sonuclar yazıldı")

