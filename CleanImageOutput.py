from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import numpy as np

#Read all fits files in specified folder
fits_file=glob.glob(r'C:\Users\EvanW\Desktop\Astro\*.fits')

#Count fits files in folder
file_count=len(fits_file)
print(file_count)

#For all files in folder, create image and histogram
#Set automatic min and max brightness values, Double checking recommended
for i in range(file_count):
    hdul=fits.open(fits_file[0])
    print(hdul.info())
    data=hdul[0].data
    print(data[i])
    header=hdul[0].header
    hdul.close()
    #Print a specific header value
    print(header['NAXIS1'])
    print('Exposure time', header['EXPTIME'],('Temp'), header['CCD-Temp'])
    expos=header['EXPTIME']
    temp=header['CCD-Temp']
    print('bin', header['YPIXSZ'])
    bins=round(header['YPIXSZ'],3)
    #print('Gain', header['GAIN'])
    #gainv=header['GAIN']
    print('Date', header['DATE-OBS'])
    datev=header['DATE-OBS']
    datev2=datev.split("T")
    print(datev2[1])
    #Set max brightness to (data max)+1%
    max_bright=np.max(data)+np.max(data)*0.01
    plt.figure(figsize=(38.4,20.4))
    plt.subplots_adjust(left=.2, right=0.8, top=0.9, bottom=0.1)
    plt.xticks(fontsize=42)
    plt.yticks(fontsize=42)
    #vmin and vmax set range of image data as function of max brightness
    plt.imshow(data, cmap='gray', vmin=max_bright/2, vmax=max_bright)
    #create variable in order to increase colorbar fontsize
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=42)
    plt.title('Astronomical Body',fontsize=75)
    plt.xlabel('Y Pixel Position',fontsize=60)
    plt.ylabel('X Pixel Position',fontsize=60)
    plt.show()
    #End of First plot
    print('Min:', np.min(data))
    print('Max:', np.max(data))
    print('Mean:', np.mean(data))
    print('Stdev:', np.std(data))
    minv=np.min(data)
    mean=np.mean(data)
    meanr=round(mean,4)
    maxv=np.max(data)
    dev=np.std(data)
    devr=round(dev,4)
    #Begin histogram plot
    print(type(data.flatten()))
    print(data.flatten().shape)
    plt.figure(figsize=(38.4,20.4))
    #Set range of histogram brightness values (x-axis)
    #This is done since most points on histogram dominated by darkness
    plt.hist(data.flatten(), bins=100, range=(max_bright/35,max_bright))
    plt.xlabel('Pixel Brightness',fontsize=60)
    plt.ylabel('Pixel Count',fontsize=60)
    plt.xticks(fontsize=42)
    plt.yticks(fontsize=42)
    #plt.locator_params(axis='both', nbins=8)
    plt.title('Distribution of Brightness',fontsize=75)
    txtbottom=(f'Min:{minv}    Max:{maxv}    Mean:{meanr}    Stdev:{devr}')
    plt.figtext(0.5, 0.01, txtbottom , wrap=True, horizontalalignment='center', fontsize=42)
    txtright=(f'Exposure:{expos}s\nTemp:{temp}c\nBins:{bins}\nGain:{gainv}\nDate:{datev2[0]}\nTime:{datev2[1]}')
    #plt.text(0, 0, txtright, fontsize = 42)
    plt.text(0.98, 0.95, txtright, transform=plt.gca().transAxes, ha='right', va='top', fontsize=42)
    plt.show()
   
    #Main header, temp, exposure time, binning gain.
    #no RA or declination, date no time, no location or altitude.