#!/usr/bin/env python
# coding: utf-8

# In[26]:


#from selenium .webdriver.common.keys import Keys
#from selenium import webdriver
from bs4 import BeautifulSoup
#import requests
import urllib
import pandas as pd
import csv
import re
import numpy as np


# In[48]:


#If you are running the int benchmark
#benchmark_type = "Int"
#If you are running the float benchmark use this
benchmark_type="Float"

if(benchmark_type=="Int"):
    benchmark_folder = "./Int"
    spec_results_file = benchmark_folder + "/Spec_INT_Results_2006.csv"
    set_cpus_file = benchmark_folder + "/Int_set_cpus.csv"
    mem_data_file = benchmark_folder + "/Int_Mem_data.csv"
    additional_cpu_world_info_file= benchmark_folder + "/Int_Additional_cpu-world_information.csv"
    spec_dataset_file = benchmark_folder + "/Int_SPEC_DATASET.csv"
    raw_bus_speed_file = benchmark_folder + "/Int_raw_bus_speed.csv"
    converted_bus_speed_file = benchmark_folder + "/Int_converted_bus_speed.csv"
    dataset_file = benchmark_folder + "/Int_dataset.csv"
    dataset_final_file = benchmark_folder + "/Int_dataset_final.csv"
    
else:
    benchmark_folder = "./Float"
    spec_results_file = benchmark_folder + "/Spec_FP_Results_2006.csv"
    set_cpus_file = benchmark_folder + "/Float_set_cpus.csv"
    mem_data_file = benchmark_folder + "/Float_Mem_data.csv"
    additional_cpu_world_info_file= benchmark_folder + "/Float_Additional_cpu-world_information.csv"
    spec_dataset_file = benchmark_folder + "/Float_SPEC_DATASET.csv"
    raw_bus_speed_file = benchmark_folder + "/Float_raw_bus_speed.csv"
    converted_bus_speed_file = benchmark_folder + "/Float_converted_bus_speed.csv"
    dataset_file = benchmark_folder + "/Float_dataset.csv"
    dataset_final_file = benchmark_folder + "/Float_dataset_final.csv"
    


# In[49]:


def download_spec_data(spec_url):
    
    r = requests.get(spec_url)
    #Create beautiful-soup object
    soup = BeautifulSoup(r.content, 'html5lib')
    #Find all links on web-page
    links = soup.findAll('a')

    for link in links:
        href = link.get('href')
        if (str(href).endswith('csv')):
            csv_links = str(href)
            # print(csv_links)
            full_url = ("http://spec.org/cpu2006/results/" + csv_links)
            # urllib.urlretrieve(full_url, file_name)
            
          
    response = urllib.urlopen(full_url)
    csv = response.read()
    csv_str = str(csv)
    lines = csv_str.split("\\n")
    dest_url = r'SPEC_INT_RESULTS_2006.csv'
    fx = open(dest_url,"a")
    print('Downloading files....')
    for line in lines:
        #print(line)
        fx.write(line + '\n')


# In[50]:


def extract_SPEC_fields_to_file(spec_file):

    #dataset_file = open(spec_file,'r')
    
    #For Python 3: 
    dataset_file = open(spec_file,'r',errors='ignore')
       
    dataset_file_reader = csv.reader(dataset_file)
    system = [0] * 2000
    arch = [0] * 2000
    clock = [0] * 2000
    l1_size = [0] * 2000
    l2_size = [0] * 2000
    l3_size = [0] * 2000
    num_of_threads = [0] * 2000
    number_of_cpus = [0] * 2000
    mem_type = [0] * 2000
    mem_clock = [0] * 2000
    mem_size = [0] * 2000
    ld_shared_by_cores = [1] * 2000
    l2_shared_by_cores = [1] * 2000


    memory1 = {}
    memory_data = {'mem-type': mem_type, 'mem-frequency': mem_clock}
    # m_df = pd.DataFrame(memory_data)
    # m_df.to_csv("Memory_Data.csv")

    memory_file = open(mem_data_file,'r')
    csv_reader = csv.reader(memory_file)

    for row in csv_reader:
        #print(row)
        memory={row[1]:row[2]}
        memory1.update(memory)
    #print(csv_reader)

    chips = 1
    index = 0

    for row in dataset_file_reader:
        if(index >= 2000):
            break

        if 'Hardware Model:' in row:
            system_name = row[1].replace(',','-')
            system[index]= system_name

        if 'CPU Name' in row:
            arch_name = row[1]
            arch[index] = arch_name
            #print(arch_name)

        if 'CPU MHz' in row:
            cpu_clock_str = float(row[1])
            cpu_clock = cpu_clock_str/1000
            clock[index] = cpu_clock

            #print(cpu_clock)
        if 'CPU(s) enabled' in row:
            #print(row[1])
            noOfCPUs = re.findall(r'\d+', row[1])
            number_of_cpus[index] = noOfCPUs[0]

            chips_array = re.search(r'(\d+) chips', row[1])

            cores_res = re.search(r'(\d+) cores', row[1])
            if(cores_res!=None):
                cores = int(cores_res.group(1))

            if chips_array is not None:
                chips=int(chips_array.group(1))

            res = re.search(r'(\d+) threads/core', row[1])

            if res is None:
                threads = cores
                num_of_threads[index] = threads
                #print(threads)

            else:
                threads_per_core = res.group(1)
                threads = int(cores) * int(threads_per_core)     
                num_of_threads[index] = threads
                #print(threads)

        if 'Primary Cache' in row:
            #print(row,type(row[1]))
            primary_cache_int = re.findall(r'\d+', row[1])
            l1d_size = int(primary_cache_int[0])
            l1_size[index] = l1d_size

            #print(l1d_size)

        if 'Secondary Cache' in row:
            l2_cache = re.findall(r'\d+', row[1])
            l2 = int(l2_cache[0])
            l2_size[index] = l2

            #print(l2_size)

        if 'L3 Cache' in row:
            l3_cache = re.findall(r'\d+', row[1])
            #print(l3_cache)
            #print(l3_cache)


            if(len(l3_cache)>0):
                l3 = int(l3_cache[0])
                l3_size[index] = l3 * 1024 * chips
            chips = 1    



        if 'Memory' in row:
            memSize = re.findall(r'\d+', row[1])
            mem_size[index] = memSize[0]
            
            memType=re.findall(r'\bPC[0-9][A-Z]?-\w+', row[1])
            #print(memType)
            if len(memType)<1:
                memType=re.findall(r'667 MHz', row[1])
                if len(memType)<1:
                    memType=re.findall(r'800 MHz', row[1])
                    if len(memType)<1:
                        memType=memType=re.findall(r'DDR400', row[1])
                #print(memType)
            if len(memType) <1:
                memType=re.findall(r'\bDDR[0-9]?-\w+', row[1])
                #print(memType)
            if len(memType)<1:
                memType=re.findall(r'\bDDR[0-9]?', row[1])
                #print(memType)
            if len(memType)<1:
                memType=re.findall(r'667 MHz', row[1])
                if len(memType)<1:
                    memType=re.findall(r'800 MHz', row[1])
                #print(memType)

            if len(memType)>0:
                #memList.append(memType[0])
                memType[0]=memType[0].replace(" ", "")
                if memType[0] in memory1.keys():
                    mem_type[index]= memType[0]
                    mem_clock[index]= memory1[memType[0]]

        #print(memory1.keys())            
            index+=1            


    spec_info = {'system_name':system, 'arch':arch, 'cpu_clock':clock,'ld_shared_by_cores':ld_shared_by_cores,'l2_shared_by_cores':l2_shared_by_cores, 'l1_size':l1_size,'l2_size':l2_size, 'l3_size': l3_size,
                'no_of_threads':num_of_threads, 'num-cpus':number_of_cpus,'mem_type':mem_type,'mem_clock':mem_clock,'mem_size':mem_size}


    spec_df = pd.DataFrame(spec_info) 
    spec_df.to_csv(spec_dataset_file, index=False)
    
    spec_df = pd.read_csv(spec_dataset_file)
    cpu_df = pd.DataFrame(pd.unique(spec_df['arch']))
    cpu_df.to_csv(set_cpus_file, header=False, index=False)


# In[51]:


def collect_additional_fields_form_cpu_world():
    
    cpu_names = []
    cpu_count = 0
    my_dataset = open(set_cpus_file,"r")
    cpu_world_info = open(additional_cpu_world_info_file,"a")
    csv_reader = csv.reader(my_dataset)
    cpu_world_info.write('CPU Name,')
    cpu_world_info.write('Bus Speed,')
    cpu_world_info.write('Bus Speed DMI,')
    cpu_world_info.write('Number of Threads,')
    cpu_world_info.write('L1 assoc Ins,')
    cpu_world_info.write('L1 assoc data,')
    cpu_world_info.write('L2 assoc,')
    cpu_world_info.write('L3 assoc,')
    cpu_world_info.write('FSB,')
    cpu_world_info.write('\n')
    for row in csv_reader:
        for names in row:
            cpu_names.append(names)
        if cpu_count < 8:
            cpu_count+=1
            print(cpu_count,names)
            cpu_world_info.write(names)
            #cpu_world_info.write(',')
    #print(cpu_names)

            driver = webdriver.Chrome('chromedriver.exe')
            #driver.get('http://www.cpu-world.com/')
            driver.get('https://www.google.com/')

            id_box = driver.find_element_by_name('q')
            #id_box = driver.find_element_by_id('PART_S')
            id_box.send_keys('cpu-world '+ cpu_names[cpu_count-1], Keys.ENTER)
            acutal_link = driver.find_element_by_partial_link_text(cpu_names[cpu_count-1])
            acutal_link.click()
            table = driver.find_elements_by_xpath("//table[@class='spec_table']")
            tr = driver.find_elements_by_xpath('//tr')
            tr_Data = [x.text for x in tr]
            #print(tr_Data)
            data = [x.text for x in table]
            #print(data,type(data))
            data_string = data[0].encode("utf-8")
            #print(data_string,type(data_string))
            data_array = data_string.split('\n')
            #print(data_array)
            info = []
            l1_flag=0
            bus_flag = 0

            for line in data_array:

                if 'Bus speed' in line:
                    #print(line)
                    bs = re.search(r"[-+]?\d*\.\d+|\d+ GT/s", line)
                    if bs is None:
                        cpu_world_info.write(',')
                    else:
                        bus_speed = bs.group()
                        print(bus_speed)
                        cpu_world_info.write(',')
                        cpu_world_info.write(bus_speed)

                if 'DMI' in line:
                    bus_flag = 1
                    print(line)
                    bs = re.search(r"[-+]?\d*\.\d+|\d+ GT/s", line)
                    bus_speed = bs.group()
                    print(bus_speed)
                    cpu_world_info.write(',')
                    cpu_world_info.write(bus_speed)

                if bus_flag == 1:
                    if 'The number of threads' in line:
                        # print(line)
                        threads = re.findall('\d+', line)
                        num_of_threads = (int(threads[0]))
                        print(type(num_of_threads))
                        cpu_world_info.write(',')
                        cpu_world_info.write(str(num_of_threads))

                if bus_flag == 0:
                    if 'The number of threads' in line:
                        # print(line)
                        threads = re.findall('\d+', line)
                        num_of_threads = (int(threads[0]))
                        print(type(num_of_threads))
                        cpu_world_info.write(',,')
                        cpu_world_info.write(str(num_of_threads))

                if 'Level 1 cache size' in line:
                    l1_flag=1
                    #print(line)

                    l1 = re.search(r'(\d+)-way',line)
                    if l1 is None:
                        cpu_world_info.write(',')
                    else:
                        l1_assoc = l1.group(1)
                        cpu_world_info.write(',')
                        cpu_world_info.write(l1_assoc)
                        print(l1_assoc)

                if l1_flag==1:
                    if 'data caches' in line:
                        assoc1 = re.search(r'(\d+)-way',line)
                        if assoc1 is None:
                            cpu_world_info.write(',')
                        else:
                            assoc1 = assoc1.group(1)
                            cpu_world_info.write(',')
                            cpu_world_info.write(assoc1)
                            print(assoc1)

                        l1_flag=0
                    else:
                        continue

                if 'Level 2 cache size' in line:
                    #print(line)
                    l2 = re.search(r'(\d+)-way',line)
                    if l2 is None:
                        cpu_world_info.write(',')
                    else:
                        l2_assoc = l2.group(1)
                        cpu_world_info.write(',')
                        cpu_world_info.write(l2_assoc)
                        print(l2_assoc)

                if 'Level 3 cache size' in line:
                    print(line)
                    l3 = re.search(r'(\d+)-way', line)
                    l3_type = type(l3)
                    if l3_type is None or l3 is None:
                        cpu_world_info.write(',')
                        cpu_world_info.write('\n')
                    else:
                        l3_assoc = l3.group(1)
                        #print(l3_assoc)
                        cpu_world_info.write(',')
                        cpu_world_info.write(l3_assoc)
                        cpu_world_info.write('\n')

            driver.close()


# In[52]:


#Add raw_bus_speed field
def add_raw_bus_speed():
    data_file = open(additional_cpu_world_info_file,'r')
    new_file = open(raw_bus_speed_file,'a')
    #new_file = open('add_rows','a')
    file_reader = csv.reader(data_file)
    cnt=0
    for row in file_reader:
        print(row[0],row[1])
        if row[1]!='' and row[2]=='' and cnt>0:
            print(row[0],row[1])
            new_file.write(row[0])
            new_file.write(',')
            new_file.write(row[1])
            new_file.write(' QPI,\n')


        if row[1]=='' and row[2]!='' and cnt>0:
            print(row[0],row[2])
            new_file.write(row[0])
            new_file.write(',')
            new_file.write(row[2])
            new_file.write(' DMI\n')


        if row[1]=='' and row[2]=='' and row[8]!='' and cnt>0:
            new_file.write(row[0])
            new_file.write(',')
            new_file.write(row[8])
            new_file.write(' FSB')
            new_file.write('\n')

        if row[1]!='' and row[2]!='' and cnt>0:
            new_file.write(row[0])
            new_file.write(',')
            new_file.write(row[1])
            new_file.write(' QPI')
            new_file.write('\n')
        cnt+=1


# In[53]:


#Converted Bus speed
def convert_bus_speed():
    file = open(raw_bus_speed_file,'r')
    con_file = open(converted_bus_speed_file,'a')
    cnt_xeons = 0
    cnt_core2 = 0
    cnt_extreme = 0
    cnt_core_duo = 0
    cnt_pentium = 0


    file_reader = csv.reader(file)
    con_file.write('arch,')
    con_file.write('raw,')
    con_file.write('converted\n')
    for row in file_reader:
        if 'QPI' in row[1]:
            if(row[1]=='6.4 QPI'):
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('25.6')
                con_file.write('\n')

            elif(row[1]=='8 QPI'):
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('32.0')
                con_file.write('\n')

            elif(row[1]=='9.6 QPI'):
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('38.4')
                con_file.write('\n')

            elif(row[1]=='4.8 QPI'):
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('19.2')
                con_file.write('\n')

            elif (row[1] == '5.86 QPI'):
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('23.44')
                con_file.write('\n')

            elif (row[1] == '7.2 QPI'):
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('28.8')
                con_file.write('\n')

            else:
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('41.6')
                con_file.write('\n')


        if 'Xeon' in row[0]  and 'FSB' in row[1]:
            cnt_xeons+=1
            find_speed_xeon = re.search(r"\d+ ", row[1])
            int_bus_speed_xeon =int(find_speed_xeon.group())
            final_bus_speed_xeon = (int_bus_speed_xeon * 8)/1024
            con_file.write(row[0])
            con_file.write(',')
            con_file.write(row[1])
            con_file.write(',')
            con_file.write(str(final_bus_speed_xeon))
            con_file.write('\n')

            #print(final_bus_speed_xeon)

        elif 'Core 2 Duo' in row[0] and 'FSB' in row[1]:
            cnt_core2+=1
            find_core_2_duo = re.search(r"\d+ ", row[1])
            int_speed = int(find_core_2_duo.group())
            final_speed_core2 = (int_speed * 8)/1024
            con_file.write(row[0])
            con_file.write(',')
            con_file.write(row[1])
            con_file.write(',')
            con_file.write(str(final_speed_core2))
            con_file.write('\n')
            #print(final_int_speed)

        elif 'Core 2 Extreme' in row[0] and 'FSB' in row[1]:
            cnt_extreme+=1
            find_core_2_extreme = re.search(r"\d+ ", row[1])
            #print(find_core_2_extreme)
            int_speed = int(find_core_2_extreme.group())
            final_int_speed_extreme = (int_speed * 8)/1024
            con_file.write(row[0])
            con_file.write(',')
            con_file.write(row[1])
            con_file.write(',')
            con_file.write(str(final_int_speed_extreme))
            con_file.write('\n')

            #print(final_int_speed)

        elif 'Core Duo' in row[0] and 'FSB' in row[1]:
            cnt_core_duo+=1
            find_core_duo = re.search(r"\d+ ", row[1])
            int_speed = int(find_core_duo.group())
            final_speed_duo = (int_speed * 4)/1024
            con_file.write(row[0])
            con_file.write(',')
            con_file.write(row[1])
            con_file.write(',')
            con_file.write(str(final_speed_duo))
            con_file.write('\n')
            #print(final_int_speed)

        elif 'Pentium' in row[0] and 'FSB' in row[1]:
            cnt_pentium+=1
            find_pentium = re.search(r"\d+ ", row[1])
            int_speed = int(find_pentium.group())
            final_speed_pentium = (int_speed * 8)/1024
            con_file.write(row[0])
            con_file.write(',')
            con_file.write(row[1])
            con_file.write(',')
            con_file.write(str(final_speed_pentium))
            con_file.write('\n')

            #print(final_int_speed)

    #print("Total FSBs:",cnt_xeons + cnt_core2 + cnt_extreme + cnt_core_duo + cnt_pentium)
        if 'DMI' in row[1]:
            #print(row[1])
            find_dmi = re.search(r"[-+]?\d*\.\d+|\d+", row[1])
            dmi_value = find_dmi.group()
            if(dmi_value == '5'):

                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('2')
                con_file.write('\n')


            elif(dmi_value == '2.5'):
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('1')
                con_file.write('\n')

            else:
                con_file.write(row[0])
                con_file.write(',')
                con_file.write(row[1])
                con_file.write(',')
                con_file.write('0.5')
                con_file.write('\n')


# In[54]:


def append_all_fields_to_dataset(cpu_world_file,spec_file):
    
    spec_df = pd.read_csv(spec_file)
    spec_cpu_names = spec_df['arch']

    cpu_world_df = pd.read_csv(cpu_world_file)
    cpu_world_names = cpu_world_df['CPU Name']


    spec_index = 0
    index = 0
    flag = 0
    bus_speed_qpi = [0]*2000
    bus_speed_dmi = [0]*2000
    l1_ins_assoc = [0]*2000
    l1_data_assoc = [0]*2000
    l2_assoc = [0]*2000
    l3_assoc = [0]*2000

    qpi = cpu_world_df['Bus Speed']
    dmi = cpu_world_df['Bus Speed DMI']
    l1_ins = cpu_world_df['L1 assoc Ins']
    l1_data = cpu_world_df['L1 assoc data']
    l2 = cpu_world_df['L2 assoc']
    l3 = cpu_world_df['L3 assoc']


    for spec_name in spec_cpu_names:
        flag = 0
        index = 0
        for cpu_name in cpu_world_names: 

            if(spec_name == cpu_name):
                flag = 1
                break
            else:
                index+=1

        if(flag == 1):  
            bus_speed_qpi[spec_index] = qpi[index]
            bus_speed_dmi[spec_index] = dmi[index]
            l1_ins_assoc[spec_index] = l1_ins[index]
            l1_data_assoc[spec_index] = l1_data[index]
            l2_assoc[spec_index] = l2[index]
            l3_assoc[spec_index] = 0 if np.isnan(l3[index]) else l3[index]
        spec_index+=1



    spec_df['bus_speed_qpi'] = bus_speed_qpi
    spec_df['bus_speed_dmi'] = bus_speed_dmi
    spec_df['l1_ins_assoc'] = l1_ins_assoc
    spec_df['l1_data_assoc'] = l1_data_assoc
    spec_df['l2_assoc'] = l2_assoc
    spec_df['l3_assoc'] = l3_assoc

    spec_df.to_csv(dataset_file, index=False)


# In[55]:


def append_bus_field_to_dataset():
    
    spec_df = pd.read_csv(dataset_file)
    spec_cpu_names = spec_df['arch']

    cpu_world_df = pd.read_csv(converted_bus_speed_file)
    cpu_world_names = cpu_world_df['arch']


    spec_index = 0
    index = 0
    flag = 0
    bus_speed1 = [0]*2000
    bus_speed2 = [0]*2000



    bs1 = cpu_world_df['raw']
    bs2 = cpu_world_df['converted']



    for spec_name in spec_cpu_names:
        flag = 0
        index = 0
        for cpu_name in cpu_world_names: 

            if(spec_name == cpu_name):
                flag = 1
                break
            else:
                index+=1

        if(flag == 1):  
            bus_speed1[spec_index] = bs1[index]
            bus_speed2[spec_index] = bs2[index]

        spec_index+=1



    spec_df['raw_bus_speed'] = bus_speed1
    spec_df['converted_bus_speed'] = bus_speed2
    
    mem_array = spec_df['mem_type']
    ddr_array = []
    for mem in mem_array:
        if('PC2' in mem or 'DDR2' in mem):
            ddr_array.append(2)
        elif('PC3L' in mem):
            ddr_array.append(3.5)
        elif('PC3' in mem or 'DDR3' in mem):
            ddr_array.append(3)
        elif('PC4' in mem or 'DDR4' in mem):
            ddr_array.append(4)
        else:
            ddr_array.append(1)
        
    spec_df['ddr_type'] = ddr_array
    
    spec_df = spec_df.drop(spec_df[(spec_df.l1_ins_assoc.astype(float) <=0) | (spec_df.l1_data_assoc.astype(float) <=0)| (spec_df.l2_assoc.astype(float)<=0) | (spec_df.converted_bus_speed.astype(float)<=0)].index)
    spec_df.dropna(axis=0, how='any', thresh=None, subset=['l1_ins_assoc', 'l1_data_assoc', 'l2_assoc'], inplace=True)

    spec_df.to_csv(dataset_final_file, index=False)


# In[56]:


def add_runtimes_to_dataset():
    

    file_names = ['400.perlbench.csv','401.bzip2.csv','403.gcc.csv','429.mcf.csv','445.gobmk.csv','456.hmmer.csv',
                  '458.sjeng.csv','462.libquantum.csv','464.h264ref.csv','471.omnetpp.csv','473.astar.csv','483.xalancbmk.csv']

    count_system_name = 1
    count_cpu_name = 0
    system_name = []
    final_list = []
    cpu_names = []
    temp_count1 = 0
    temp_count2 = 0
    temp_count3 = 0
    temp_count4 = 0
    x=0
    my_run_time = []
    temp = []
    flag = 0

    spec_df = pd.read_csv(dataset_final_file)
    #for python 3
    my_csv = open(spec_results_file,errors='ignore')
    #my_csv = open('SPEC_INT_RESULTS_2006.csv')
    csv_reader = csv.reader(my_csv)

    runtime=[[] for _ in range(12)]
    print(len(runtime))
    entries = 0
    
    system_names_to_search= spec_df['system_name']
    
    all_system_names=[]
    
    runtime_info=[]
    
    for row in csv_reader:
        size = len(row)

        #RUN-TIME INFORMATION
        if 'Hardware Model:' in row:
            system_name = row[1].replace(',','-')
            all_system_names.append(system_name)
            temp.append(system_name)
            runtime_info.append(temp)
            temp=[]
        
        if size == 1:
            if 'Selected Results Table' in row[0]:
                flag = 1
                count = 0
            if 'Full Results Table' in row[0]:
                flag = 0
        if size > 2 and flag == 1:
            #print(run_info[2])#Prints both the tables of base run-time
            b_r_t = re.search(r'[-+]?\d*\.\d+|\d+',row[2])# Run-Time

            if(b_r_t!=None and count<12):
                run_time = b_r_t.group()
                temp.append([row[0],run_time])
                #print(count,run_info[0],run_time)
                count += 1


    # print(len(runtime1))
    #print(my_run_time)
    print(len(system_names_to_search))
    print(len(all_system_names))
    print(len(runtime_info))
    
    for sys_name in system_names_to_search:
        index=0
        for name in all_system_names:
            if(sys_name in name):
                break
            else:
                index+=1

        if(len(runtime_info[index])<=1):
            for x in range(0,12):
                runtime[x].append(0)
        else:
            for x in range(0,12):
                runtime[x].append(runtime_info[index][x][1])
                    
    print(len(runtime[0]))
    for x in range(0,12):
        spec_df_copy = spec_df.copy()
        spec_df_copy['runtime'] = runtime[x]
        print (spec_df_copy['runtime'])
        spec_df_copy = spec_df_copy.drop(spec_df_copy[(spec_df_copy.runtime == 0) | (spec_df_copy.runtime == '64')].index)
        spec_df_copy.to_csv(benchmark_folder + '/' + runtime_info[0][x][0]+'.csv', index=False)


# In[57]:


if __name__ == "__main__":
     
    #download_spec_data('http://spec.org/cpu2006/results/cint2006.html')
    extract_SPEC_fields_to_file(spec_results_file)
    #collect_additional_fields_form_cpu_world()
    add_raw_bus_speed()
    convert_bus_speed()
    append_all_fields_to_dataset(additional_cpu_world_info_file, spec_dataset_file)
    append_bus_field_to_dataset()    
    add_runtimes_to_dataset()
    
    


# In[ ]:





# In[ ]:




