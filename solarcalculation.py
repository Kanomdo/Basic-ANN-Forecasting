import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def SAlSAzCal(timestamp):
    StdLong = -105
    LcLong = -100.5
    LcLat = 14.67
    n = timestamp.timetuple().tm_yday
    B = (360*(n-81))/364
    E = 9.87*np.sin(2*B*(np.pi/180))-7.53 * \
        np.cos(B*(np.pi/180))-1.5*np.sin(B*(np.pi/180))
    StdTime = timestamp.hour + timestamp.minute/60 + timestamp.second/3600
    SolarTime = StdTime + (4*(StdLong-LcLong) + E)/60
    w = 15*(SolarTime-12)
    SDec = 23.45*np.sin((np.pi/180)*(360*(284+n))/365)
    # SolarAltitudeAngle
    SAl = np.arcsin(np.cos(LcLat*(np.pi/180))*np.cos(SDec*(np.pi/180)) *
                    np.cos(w*(np.pi/180))+np.sin(LcLat*(np.pi/180))*np.sin(SDec*(np.pi/180)))
    SAl = SAl*(180/np.pi)
    if SAl < 0:
        SAl = 0
    Temp1 = ((np.cos(SDec*(np.pi/180)))*np.sin(w*(np.pi/180))) / \
        np.cos(SAl*(np.pi/180))
    SAz = (np.arctan(Temp1/np.sqrt(1 - (Temp1**2))))
    SAz1 = SAz*(180/np.pi)
    if SAz1 > 0:
        SAz2 = 180-(SAz1)
    else:
        SAz2 = -(180-abs(SAz1))
    if SDec > LcLat:
        SAz3 = SAz2
    else:
        SAz3 = SAz1
    SAzimuth = SAz3
    return [SAl, SAzimuth]


def SolarAngle(dataframe):
    timestamp = dataframe["timestamp"]
    SAl, SAzimuth = SAlSAzCal(timestamp)
    SAl_next_hr, SAz_next_hr = SAlSAzCal(timestamp + timedelta(hours=1))
    return SAl, SAl_next_hr, SAzimuth, SAz_next_hr


def InciAng(PAz, SAl, SAz):
    alpha_s = SAl*(np.pi/180)
    beta = 90*(np.pi/180)
    gamma_s = SAz*(np.pi/180)
    gamma_p = PAz*(np.pi/180)
    SInci = np.arccos(np.sin(alpha_s) * np.cos(beta) + np.cos(alpha_s) * np.sin(gamma_s) * np.sin(
        beta) * np.sin(gamma_p) + np.cos(alpha_s) * np.cos(gamma_s) * np.sin(beta) * np.cos(gamma_p))
    SInci = SInci*(180/np.pi)
    return SInci
