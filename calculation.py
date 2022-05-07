import numpy as np
import pandas as pd
from solarcalculation import SolarAngle, InciAng

def AngleCalculation(df):
    df_SAlSAz = pd.DataFrame(df.index).apply(SolarAngle, axis=1)
    df[['SAl', 'SAlNextHr', 'SAz', 'SAzNextHr']] = pd.DataFrame(
        df_SAlSAz.tolist(), index=df.index)
    df['SInN'] = InciAng(180, df['SAl'], df['SAz'])
    df['SInE'] = InciAng(-90, df['SAl'], df['SAz'])
    df['SInS'] = InciAng(0, df['SAl'], df['SAz'])
    df['SInW'] = InciAng(90, df['SAl'], df['SAz'])
    df['SInNNextHr'] = InciAng(180, df['SAlNextHr'], df['SAzNextHr'])
    df['SInENextHr'] = InciAng(-90, df['SAlNextHr'], df['SAzNextHr'])
    df['SInSNextHr'] = InciAng(0, df['SAlNextHr'], df['SAzNextHr'])
    df['SInWNextHr'] = InciAng(90, df['SAlNextHr'], df['SAzNextHr'])
    df.loc[df['SAl'] <= 0, ['SInNNextHr',
                            'SInENextHr', 'SInSNextHr', 'SInWNextHr']] = 0
    df['SZe'] = 90 - df['SAl']
    df['SZeNextHr'] = 90 - df['SAlNextHr']
    return df

def FeatureCalculation(df):
    df.loc[df['SAl'] <= 0, ["Eeg", "Eed", "Eeb",
                            "EeN", "EeE", "EeS", "EeW", "SAl"]] = 0
    for j in ["Eeg", "Eed", "Eeb", "EeN", "EeE", "EeS", "EeW", "SAl"]:
        df[j] = df[j].values.clip(min=0)
    SolarAlRad = df['SAl']*(np.pi/180)
    EegCalNor = -36.78*np.power(SolarAlRad, 5)+188*np.power(SolarAlRad, 4)-375.95*np.power(
        SolarAlRad, 3)+306.2*np.power(SolarAlRad, 2)+15.47*SolarAlRad+0.83
    EegCalNor = EegCalNor*(1000/100)
    df['NormalizedEeg'] = df['Eeg']/EegCalNor
    df['EegMathCal'] = df['Eed'] + \
        (1.15*df['Eeb']*np.sin(df['SAl']*(np.pi/180)))
    return df

def VerticalCalculation(df):
    Eeg = df['Eeg']
    Eed = df['Eed']
    phi = df['SZe']*(np.pi/180)
    Eeb = (Eeg-Eed)/np.cos(phi)
    epsilon = (((Eed+Eeb)/Eed)+1.041*(phi**3))/(1+1.041*(phi**3))
    epsilon[Eeg <= 20] = None
    df_COE = pd.read_csv("coefficients.csv", header=0)
    df_bin = pd.DataFrame([], index=df.index)
    df_bin['epsilon'] = epsilon
    df_bin['Upperlimit'] = epsilon
    df_bin['Upperlimit'] = df_bin['Upperlimit'].apply(
        lambda x: 1.065 if x <= 1.065 else x)
    df_bin['Upperlimit'] = df_bin['Upperlimit'].apply(
        lambda x: 1.230 if x > 1.065 and x <= 1.230 else x)
    df_bin['Upperlimit'] = df_bin['Upperlimit'].apply(
        lambda x: 1.500 if x > 1.230 and x <= 1.500 else x)
    df_bin['Upperlimit'] = df_bin['Upperlimit'].apply(
        lambda x: 1.950 if x > 1.500 and x <= 1.950 else x)
    df_bin['Upperlimit'] = df_bin['Upperlimit'].apply(
        lambda x: 2.800 if x > 1.950 and x <= 2.800 else x)
    df_bin['Upperlimit'] = df_bin['Upperlimit'].apply(
        lambda x: 4.500 if x > 2.800 and x <= 4.500 else x)
    df_bin['Upperlimit'] = df_bin['Upperlimit'].apply(
        lambda x: 6.2 if x > 4.500 and x <= 6.2 else x)
    df_bin['Upperlimit'] = df_bin['Upperlimit'].apply(
        lambda x: "-" if x > 6.2 else x)
    df_bin['Upperlimit'] = df_bin['Upperlimit'].astype('str')
    df_bin['Upperlimit'][df_bin['Upperlimit'] == 'nan'] = None
    df_bin = df_bin.merge(df_COE, how="left", on='Upperlimit')
    df_bin.index = df.index
    bright_idx = (1/np.cos(phi))*(Eed/1367)
    df_bin['bright_idx'] = bright_idx
    df_bin['F1'] = df_bin['F11']+(df_bin['F12']*bright_idx)+(df_bin['F13']*phi)
    df_bin['F2'] = df_bin['F21']+(df_bin['F22']*bright_idx)+(df_bin['F23']*phi)
    df_bin['A1_North'] = df['SInN'].apply(
        lambda x: max(0, np.cos(x*np.pi/180)))
    df_bin['A1_East'] = df['SInE'].apply(
        lambda x: max(0, np.cos(x*np.pi/180)))
    df_bin['A1_South'] = df['SInS'].apply(
        lambda x: max(0, np.cos(x*np.pi/180)))
    df_bin['A1_West'] = df['SInW'].apply(
        lambda x: max(0, np.cos(x*np.pi/180)))
    df_bin['A2'] = df['SZe'].apply(lambda x: max(
        np.cos(85*(np.pi/180)), np.cos(x*(np.pi/180))))
    df_bin['EengN'] = df['Eed']*((1-df_bin['F1'])/2+((df_bin['A1_North']/df_bin['A2'])
                                 * df_bin['F1'])+df_bin['F2'])+(df_bin['A1_North']*Eeb)
    df_bin['EengE'] = df['Eed']*((1-df_bin['F1'])/2+(
        (df_bin['A1_East']/df_bin['A2'])*df_bin['F1'])+df_bin['F2'])+(df_bin['A1_East']*Eeb)
    df_bin['EengS'] = df['Eed']*((1-df_bin['F1'])/2+((df_bin['A1_South']/df_bin['A2'])
                                 * df_bin['F1'])+df_bin['F2'])+(df_bin['A1_South']*Eeb)
    df_bin['EengW'] = df['Eed']*((1-df_bin['F1'])/2+(
        (df_bin['A1_West']/df_bin['A2'])*df_bin['F1'])+df_bin['F2'])+(df_bin['A1_West']*Eeb)
    df_bin.fillna(0, inplace=True)
    return pd.concat([df.loc[:, ["EeN", "EeE", "EeS", "EeW"]], df_bin.loc[:, ['EengN', 'EengE', 'EengS', 'EengW']]], axis=1)