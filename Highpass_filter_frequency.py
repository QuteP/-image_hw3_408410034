import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from matplotlib.widgets import Slider,RadioButtons
from pic2str import aloej,churchj,housej,kitchenj
import base64

Xn=0.412453+0.357580+0.180423 #看課本有沒有
Yn=1.0
Zn=0.019334+0.119193+0.950227

def HistogramEqual(img):
    height,width=img.shape
    cnt_dic={}#nk
    s_dic={}#還沒round的sk
    s_dic[-1]=0
    for i in range(0,256):
        cnt_dic[i]=0
        s_dic[i]=0
    prob_lt=[0]*256#pr(rk)
    for row in img:
        for ele in row:
            cnt_dic[ele]+=1
    for i in range(0,256):
        prob_lt[i]=cnt_dic[i]/img.size
    for i in range(0,256):
        s_dic[i]=s_dic[i-1]+255*prob_lt[i]
    new_img=np.arange(img.size,dtype=np.uint8).reshape((height,width))
    for i in range(0,height):
        for j in range(0,width):
            new_img[i][j]=round(s_dic[img[i][j]])
    return new_img

def RGBToXYZ(bgr_img):
    img=bgr_img.astype(np.float)/255

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
    bgr_img[:,:,0]=b_array*255
    bgr_img[:,:,1]=g_array*255
    bgr_img[:,:,2]=r_array*255
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
    L_array=lab_img[:,:,0]
    a_array=lab_img[:,:,1]
    b_array=lab_img[:,:,2]
    lab_f_y=(L_array+16)/116
    lab_f_x=(a_array)/500+lab_f_y
    lab_f_z=lab_f_y-(b_array)/200
    x_array=inv_lab_f(lab_f_x)*Xn
    y_array=inv_lab_f(lab_f_y)*Yn
    z_array=inv_lab_f(lab_f_z)*Zn
    return (x_array,y_array,z_array)

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
    r_array=rgb_array[:,:,2]
    g_array=rgb_array[:,:,1]
    b_array=rgb_array[:,:,0]
    for h in range(0,height):
        for w in range(0,width):
            if 0<=h_array[h][w]<120:
                b_array[h][w]=i_array[h][w]*(1-s_array[h][w])
                r_array[h][w]=i_array[h][w]*(1+(s_array[h][w]*np.cos(np.deg2rad(h_array[h][w]))/np.cos(np.deg2rad(60-h_array[h][w]))))
                g_array[h][w]=3*i_array[h][w]-(r_array[h][w]+b_array[h][w])
            elif 120<=h_array[h][w]<240:
                h_array[h][w]=h_array[h][w]-120
                r_array[h][w]=i_array[h][w]*(1-s_array[h][w])
                b_array[h][w]=3*i_array[h][w]-(r_array[h][w]+g_array[h][w])
                g_array[h][w]=i_array[h][w]*(1+(s_array[h][w]*np.cos(np.deg2rad(h_array[h][w]))/np.cos(np.deg2rad(60-h_array[h][w]))))
            elif 240<=h_array[h][w]<=360:
                h_array[h][w]=h_array[h][w]-240
                g_array[h][w]=i_array[h][w]*(1-s_array[h][w])
                r_array[h][w]=3*i_array[h][w]-(b_array[h][w]+g_array[h][w])
                b_array[h][w]=i_array[h][w]*(1+(s_array[h][w]*np.cos(np.deg2rad(h_array[h][w]))/np.cos(np.deg2rad(60-h_array[h][w]))))
    return np.clip(rgb_array*255,0,255).astype(np.uint8)

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
    lab_img[:,:,0]=L_array
    lab_img[:,:,1]=a_array
    lab_img[:,:,2]=b_array
    return lab_img

def bgr_min(bgr_img):
    height,width,channel=bgr_img.shape
    min_img=np.zeros((height,width),dtype=np.float)
    for h in range(0,height):
        for w in range(0,width):
            min_img[h][w]=bgr_img[h][w].min()
    return min_img

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
    pos_num=x_pos+1+2*(y_pos-1)
    ax = axisartist.Subplot(fig, 3,2,pos_num)
    fig.add_subplot(ax)
    ax_lt.append(ax)
    ax.imshow(data,cmap='gray')
    ax.axis["bottom"].set_visible(False)
    ax.axis["left"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    plt.tight_layout()

aloe=cv2.imread('./aloe.jpg',cv2.IMREAD_UNCHANGED)
if type(aloe)==type(None):
    aloe=ReadByteImg(aloej)
church=cv2.imread('./church.jpg',cv2.IMREAD_UNCHANGED)
if type(church)==type(None):
    church=ReadByteImg(churchj)
kitchen=cv2.imread('./kitchen.jpg',cv2.IMREAD_UNCHANGED)
if type(kitchen)==type(None):
    kitchen=ReadByteImg(kitchenj)
house=cv2.imread('./house.jpg',cv2.IMREAD_UNCHANGED)
if type(house)==type(None):
    house=ReadByteImg(housej)

fig = plt.figure()
img_lt=[aloe,church,kitchen,house]
ax_lt=[]
for i,(img) in enumerate(img_lt):
    new_img=HSIToRGB(RGBToHSI(img))
    cv2.imshow("new_img",new_img)
    cv2.waitKey()
    # PlotHisto(fig,img,ax_lt,i,1)
    # PlotHisto(fig,new_img,ax_lt,i,2)


# axcolor = 'lightgoldenrodyellow'
# om = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
# som = Slider(om, r'k', 0.0, 100.0, valinit=20.0)
# def update(val):
#     s = som.val
#     cnt=0
#     for i in range(1,len(ax_lt),2):
#         ax_lt[i].imshow(HighpassTrans(img_lt[cnt],k=s),cmap='gray')
#         cnt+=1
# som.on_changed(update)
# plt.tight_layout()
# plt.show()
