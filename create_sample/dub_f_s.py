#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from sys import argv
import sys
import argparse
def parse_arg():
    parser = argparse.ArgumentParser(
            description = 'Dublicate foundation and editor',
            epilog = "@lrikozavr"
            )
    parser.add_argument(
            '--filename','-fi', dest = 'filename', 
            type = str, help = 'File name'
            )
    parser.add_argument(
            '--flag','-fl', dest = 'flag',
            default = "c", type = str, help = '''
            Flag:
            c Only count of dublicate;                 
            f extract non dublicate in file not_dub.txt;
            fa extract only -//-;                      
            cf both of -c and -f;                      
            d extract only dublecate;                  
            cd both of -c and -d'''
            )
    parser.add_argument(
            '--header','-he',dest = 'header',
            default = "N", type = str, help = 'YN/YY/N header'            
            )
    parser.add_argument(
            '--separator','-s', dest = 'separator',
            default = ",", type = str, help = 'Separator symbol like \t or ,'
            )
    parser.add_argument(
            '--columns','-col', dest = 'columns', type = int,
            default = [1,2], nargs=2, help = 'Column of potential dublicate' 
            )
    return parser.parse_args()

args=parse_arg()
'''
if (len(argv)<2):
    print ("-c Only count of dublicate")
    print ("-f extract non dublicate in file not_dub.txt")
    print ("-fa extract only -//-")
    print ("-cf both of -c and -f")
    print ("-d extract only dublecate")
    print ("-cd both of -c and -d")
    print ("-t symvol of separator")
    print ("Example: dublicate_fnd.py -c path/filename.csv path/filename.csv")
    print ("Example: dublicate_fnd.py -t , -c path/filename.csv path/filename.csv")
    exit()
else:
    j = argv[1]
    G = argv[2]
    if (j!="-c"):
    	J = argv[3]
      	
#############        
dublicate_fnd.py -c path/filename.csv
-c Only count of dublicate
-f extract non dublicate in file not_dub.txt
-fa extract only -//-
-cf both of -c and -f
-d extract only dublecate
-cd both of -c and -d
'''
#
i=1
k=0
z=0
split_sym=args.separator
j=args.flag
G=args.filename

gc1=args.columns[0]-1
gc2=args.columns[1]-1
header=args.header

'''
with open(G) as fileobject:
    for line in fileobject:
        ....
'''
if (j=="c"):
    for line in open(G):
        if (not header or header=="N"):
            n=line.split(split_sym)    
            if (i>1):
                if (decn==n[gc2]) and (ran==n[gc1]):
                    k+=1
                    z+=1
                else:
                    if (z>1):
                        k+=1
                    ran=n[gc1]
                    decn=n[gc2]
                    z=1
            else:
                ran=n[gc1]
                decn=n[gc2]
                i=2
                z=1
        else:
            if (header=="YY"):
                print(str(line),end='')
            header="N"
    if (z>1):
        k+=1
    fs="Count of dublicate----" + str(k) + "\n"
    sys.stderr.write(fs)
elif (j=="f"):
    for line in open(G):   
        if (not header or header=='N'):
            n=line.split(split_sym)    
            if (i>1):
                if (decn!=n[gc2]) or (ran!=n[gc1]):
                    ran=n[gc1]
                    decn=n[gc2]
                    print(str(line), end='')
            else:
                ran=n[gc1]
                decn=n[gc2]
                i=2
                print(str(line), end='')
        else:
            if (header=="YY"):
                print(str(line),end='')
            header="N"
elif (j=="cf"):
    for line in open(G):   
        if (not header or header=='N'):
            n=line.split(split_sym)    
            if (i>1):
                if (decn==n[gc2]) and (ran==n[gc1]):
                    k=k+1
                else:
                    ran=n[gc1]
                    decn=n[gc2]
                    print(str(line), end='')
            else:
                ran=n[gc1]
                decn=n[gc2]
                i=2
                print(str(line), end='')
        else:
            if (header=="YY"):
                print(str(line),end='')
            header="N"
    fs="Count of dublicate----"+str(k)+"\n"
    sys.stderr.write(fs)
elif (j=="d"):
    k=1
    l=""
    for line in open(G): 
        if (not header or header=='N'):
            n=line.split(split_sym)    
            if (i>1):
                if (decn==n[gc2]) and (ran==n[gc1]):
                    print(str(l), end='')
                    l=str(line)
                    k=k+1
                else:
                    if (k>1):
                        print(str(l), end='')
                    ran=n[gc1]
                    decn=n[gc2]
                    l=str(line)
                    k=1
            else:
                ran=n[gc1]
                decn=n[gc2]
                l=str(line)
                i=2
        else:
            if (header=="YY"):
                print(str(line),end='')
            header="N"
    if (k>1):
        print(str(l), end='')
elif (j=="fa"):
    l=""
    for line in open(G):
        if (not header or header=='N'):
            n=line.split(split_sym)    
            if (i>1):
                if (decn!=n[gc2]) or (ran!=n[gc1]):
                    ran=n[gc1]
                    decn=n[gc2]
                    if (z==1):
                        print(str(l), end='')
                    l=str(line)
                    z=1
                else:
                    z+=1
            else:
                ran=n[gc1]
                decn=n[gc2]
                i=2
                z=1
                l=str(line)
        else:
            if (header=="YY"):
                print(str(line),end='')
            header="N"
    if (z==1):
        print(str(l), end='')
elif (j=="cd"):
    k=1
    count=0
    l=""
    for line in open(G):  
        if (not header or header=='N'):
            n=line.split(split_sym)
            if (i>1):
                if (decn==n[gc2]) and (ran==n[gc1]):
                    print(str(l), end='')
                    k+=1
                    count+=1
                else:
                    if (k>1):
                        print(str(l), end='')
                        count+=1
                    ran=n[gc1]
                    decn=n[gc2]
                    l=str(line)
                    k=1
            else:
                ran=n[gc1]
                decn=n[gc2]
                l=str(line)
                i=2
        else:
            if (header=="YY"):
                print(str(line),end='')
            header="N"
    if (k>1):
        print(str(l), end='')
        count+=1
    fs="Count of dublicate----" + str(count) + "\n"
    sys.stderr.write(fs)
