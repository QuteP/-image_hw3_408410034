import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from matplotlib.widgets import Slider,RadioButtons
from pic2str import aloej
import base64

Xn=0.412453+0.357580+0.180423 #看課本有沒有
Yn=1.0
Zn=0.019334+0.119193+0.950227

def GammaTrans(img,c=255,gamma=0.5):
    return np.clip(c*(np.power(img/float(np.max(img)),gamma)),0,255)

def HistogramEqual(img):
    height,width=img.shape
    value_dic={}
    for h in range(0,height):
        for w in range(0,width):
            value_dic[img[h][w]]=0
    for h in range(0,height):
        for w in range(0,width):
            value_dic[img[h][w]]+=1
    new_lt=sorted(value_dic.items())
    cdf=0.0
    new_img=np.zeros((height,width),dtype=np.float)
    for tp in new_lt:
        key,v=tp
        cdf+=v/(height*width)
        value_dic[key]=cdf
    # print(value_dic)
    for h in range(0,height):
        for w in range(0,width):
            new_img[h][w]=value_dic[img[h][w]]
    return np.clip(255*new_img,0,255)

def BGRToRGB(bgr_img):
    rgb_array=np.zeros(bgr_img.shape,dtype=np.uint8)
    rgb_array[:,:,0]=bgr_img[:,:,2]
    rgb_array[:,:,1]=bgr_img[:,:,1]
    rgb_array[:,:,2]=bgr_img[:,:,0]
    return rgb_array

def RGBToXYZ(bgr_img):
    img=np.clip(bgr_img.astype(np.float)/255,0,1)

    b_array=img[:,:,0]
    g_array=img[:,:,1]
    r_array=img[:,:,2]

    x_array=0.412453*r_array+0.357580*g_array+0.180423*b_array
    y_array=0.212671*r_array+0.715160*g_array+0.072169*b_array
    z_array=0.019334*r_array+0.119193*g_array+0.950227*b_array
    return (x_array,y_array,z_array)

def XYZToRGB(tp):
    x_array,y_array,z_array=tp
    r_array=3.240479*x_array-1.53715*y_array-0.498535*z_array
    g_array=-0.969256*x_array+1.875991*y_array+0.041556*z_array
    b_array=0.055648*x_array-0.204043*y_array+1.057311*z_array
    height,width=x_array.shape
    bgr_img=np.zeros((height,width,3),dtype=np.float)
    bgr_img[:,:,0]=np.clip(r_array*255,0,255)
    bgr_img[:,:,1]=np.clip(g_array*255,0,255)
    bgr_img[:,:,2]=np.clip(b_array*255,0,255)
    # cv2.imshow("lab_img",bgr_img.astype(np.uint8))
    # cv2.waitKey()
    return bgr_img.astype(np.uint8)

def inv_lab_f(lab_f):
    height,width=lab_f.shape
    lab_f_1=lab_f**3
    lab_f_2=(lab_f-(4/29))/((1/3)*((29/6)**2))
    arr=np.zeros((height,width),dtype=np.float)
    for h in range(0,height):
        for w in range(0,width):
            if lab_f_1[h][w]>((6/29)**3):
                arr[h][w]=lab_f_1[h][w]
            else:
                arr[h][w]=lab_f_2[h][w]
    return arr

def LabToXYZ(lab_img):
    L_array=lab_img[:,:,0]*100/255
    a_array=lab_img[:,:,1]
    b_array=lab_img[:,:,2]
    lab_f_y=(L_array+16)/116
    lab_f_x=(a_array-128)/500+lab_f_y
    lab_f_z=lab_f_y-(b_array-128)/200
    x_array=inv_lab_f(lab_f_x)*Xn
    y_array=inv_lab_f(lab_f_y)*Yn
    z_array=inv_lab_f(lab_f_z)*Zn
    return (x_array,y_array,z_array)

def lab_f(arr):
    height,width=arr.shape
    f_arr_1=arr**(1/3)
    f_arr_2=(1/3)*((29/6)**2)*arr+(4/29)
    f_arr=np.zeros((height,width),dtype=np.float)
    for h in range(0,height):
        for w in range(0,width):
            if arr[h][w]>((6/29)**3):
                f_arr[h][w]=f_arr_1[h][w]
            else:
                f_arr[h][w]=f_arr_2[h][w]
    return f_arr

def XYZToLab(tp):
    x_array,y_array,z_array=tp
    height,width=x_array.shape
    L_array=116*lab_f(y_array/Yn)-16
    a_array=500*(lab_f(x_array/Xn)-lab_f(y_array/Yn))
    b_array=200*(lab_f(y_array/Yn)-lab_f(z_array/Zn))
    lab_img=np.zeros((height,width,3),dtype=np.float)
    lab_img[:,:,0]=np.clip(L_array*255/100,0,255)
    lab_img[:,:,1]=np.clip(a_array+128,0,255)
    lab_img[:,:,2]=np.clip(b_array+128,0,255)
    return lab_img

def bgr_min(bgr_img):
    height,width,channel=bgr_img.shape
    min_img=np.zeros((height,width),dtype=np.float)
    for h in range(0,height):
        for w in range(0,width):
            min_img[h][w]=bgr_img[h][w].min()
    return min_img

def inv_cos(bgr_img):
    height,width,channel=bgr_img.shape
    b_array=bgr_img[:,:,0]
    g_array=bgr_img[:,:,1]
    r_array=bgr_img[:,:,2]
    f_x=(r_array-(g_array+b_array)/2)/(((r_array-g_array)**2+(r_array-b_array)*(g_array-b_array))**(1/2)+0.000001)
    theta=180*np.arccos(f_x)/np.pi
    h_array=np.zeros((height,width),dtype=np.float)
    for h in range(0,height):
        for w in range(0,width):
            if b_array[h][w]<=g_array[h][w]:
                h_array[h][w]=theta[h][w]
            else:
                h_array[h][w]=360-theta[h][w]
    return h_array

def HSIToRGB(hsi_img):
    i_array=hsi_img[:,:,2]/255
    s_array=hsi_img[:,:,1]/255
    h_array=hsi_img[:,:,0]/255*360
    height,width,channel=hsi_img.shape
    rgb_array=np.zeros((height,width,channel),dtype=np.float)
    r_array=rgb_array[:,:,0]
    g_array=rgb_array[:,:,1]
    b_array=rgb_array[:,:,2]
    for h in range(0,height):
        for w in range(0,width):
            if 0<=h_array[h][w]<120:
                b_array[h][w]=i_array[h][w]*(1-s_array[h][w])
                r_array[h][w]=i_array[h][w]*(1+(s_array[h][w]*np.cos(np.deg2rad(h_array[h][w]))/(np.cos(np.deg2rad(60-h_array[h][w]))+0.000001)))
                g_array[h][w]=3*i_array[h][w]-(r_array[h][w]+b_array[h][w])
            elif 120<=h_array[h][w]<240:
                h_array[h][w]=h_array[h][w]-120
                r_array[h][w]=i_array[h][w]*(1-s_array[h][w])
                b_array[h][w]=3*i_array[h][w]-(r_array[h][w]+g_array[h][w])
                g_array[h][w]=i_array[h][w]*(1+(s_array[h][w]*np.cos(np.deg2rad(h_array[h][w]))/(np.cos(np.deg2rad(60-h_array[h][w]))+0.000001)))
            elif 240<=h_array[h][w]<=360:
                h_array[h][w]=h_array[h][w]-240
                g_array[h][w]=i_array[h][w]*(1-s_array[h][w])
                r_array[h][w]=3*i_array[h][w]-(b_array[h][w]+g_array[h][w])
                b_array[h][w]=i_array[h][w]*(1+(s_array[h][w]*np.cos(np.deg2rad(h_array[h][w]))/(np.cos(np.deg2rad(60-h_array[h][w]))+0.000001)))
    return np.clip(rgb_array*255,0,255).astype(np.uint8)

def RGBToHSI(bgr_img):
    height,width,channel=bgr_img.shape
    img=bgr_img/255
    temp_sum=img[:,:,0]+img[:,:,1]+img[:,:,2]
    h_array=(inv_cos(img)/360)*255
    i_array=((temp_sum)/3)*255
    s_array=(1-3*(bgr_min(img)/(temp_sum+0.000001)))*255
    hsi_img=np.zeros((height,width,channel),dtype=np.float)
    hsi_img[:,:,0]=h_array
    hsi_img[:,:,1]=s_array
    hsi_img[:,:,2]=i_array
    # cv2.imshow("hsi",hsi_img.astype(np.uint8))
    # cv2.waitKey()
    # print(hsi_img)
    return hsi_img


def ReadByteImg(func):
    byte_data = base64.b64decode(func)
    img_buffer_numpy = np.frombuffer(byte_data, dtype=np.uint8) 
    return cv2.imdecode(img_buffer_numpy, cv2.IMREAD_UNCHANGED)

def PlotHisto(fig,data,ax_lt,x_pos,y_pos):
    pos_num=x_pos+y_pos
    ax = axisartist.Subplot(fig, 2,2,pos_num)
    fig.add_subplot(ax)
    ax_lt.append(ax)
    ax.imshow(data,cmap='gray')
    ax.axis["bottom"].set_visible(False)
    ax.axis["left"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    plt.tight_layout()

def GenerateLab(rgb_img):
    lab_img=XYZToLab(RGBToXYZ(rgb_img))
    lab_img[:,:,0]=GammaTrans(lab_img[:,:,0])
    return XYZToRGB(LabToXYZ(lab_img))

def GenerateHSI(rgb_img):
    hsi_img=RGBToHSI(rgb_img)
    hsi_img[:,:,2]=GammaTrans(hsi_img[:,:,2],gamma=0.75)
    return HSIToRGB(hsi_img)

def GenerateRGB(rgb_img,channel=3):
    new_img=np.zeros(rgb_img.shape,dtype=np.float)
    if channel==1:
        new_img=GammaTrans(rgb_img,gamma=0.6)
        # new_img=HistogramEqual(rgb_img)
    elif channel==3:
        new_img[:,:,0]=GenerateRGB(rgb_img[:,:,2],channel=1)
        new_img[:,:,1]=GenerateRGB(rgb_img[:,:,1],channel=1)
        new_img[:,:,2]=GenerateRGB(rgb_img[:,:,0],channel=1)
    return np.clip(new_img,0,255).astype(np.uint8)

aloe=cv2.imread('./aloe.jpg')
if type(aloe)==type(None):
    aloe=ReadByteImg(aloej)

fig = plt.figure()
img_lt=[aloe]
ax_lt=[]
for i,(img) in enumerate(img_lt):
    lab_img=GenerateLab(img)
    rgb_img=GenerateRGB(img)
    hsi_img=GenerateHSI(img)
    PlotHisto(fig,BGRToRGB(img),ax_lt,i,1)
    PlotHisto(fig,rgb_img,ax_lt,i,2)
    PlotHisto(fig,lab_img,ax_lt,i,3)
    PlotHisto(fig,hsi_img,ax_lt,i,4)

plt.tight_layout()
plt.show()
