# -*- coding: utf-8 -*-
"""
Getting District Data Features from U.S. Census and other sources.

"""

import json
import requests
import pandas as pd

'''
Below are the variables to be requested from Census Bureau that will provide
details for each congressional district that will allow us to find similar
districts.  The variables are divided into three categories:

Size: population, diversity
Economics: median income, median home value, median rent
Education: percent high school, percent bachelors

Full Variables List: http://api.census.gov/data/2013/acs3/variables.html
    WARNING: The variables file is very large (19.2mb)

'''

fullpop = 'B01001_001E'  # Total Population
white = 'B01001A_001E'   # White residents
income = 'B19013_001E'   # Median Household Income
homeval = 'B25075_001E'  # Median Home Value, Owner-occupied
rentval = 'B25111_001E'  # Median Rent

inform = '%s,%s,%s,%s,%s,%s' % (fullpop, income, white, homeval, rentval)

key = '923b26637864a05dcb279ee53313d95de7b4c590'    # API Key

url = 'http://api.census.gov/data/2013/acs3?get=NAME,%s&for=congressional+district:*&key=%s' % (inform, key)
request = requests.get(url)
census = json.loads(request.text)

'''
Converts the returned data into a Dataframe
'''

column_names = ['fulldistrict','state','population','income','popwhite','homeval','rentval']
index = range(len(census))
df = pd.DataFrame(columns=column_names, index=index)
i = 0
for result in census:
    df.loc[i]['fulldistrict'] = result[0]
    df.loc[i]['state'] = result[6]
    df.loc[i]['population'] = result[1]
    df.loc[i]['income'] = result[2]
    df.loc[i]['popwhite'] = result[3]
    df.loc[i]['popblack'] = result[4]
    df.loc[i]['homeval'] = result[4]
    df.loc[i]['rentval'] = result[5]
    i += 1

'''
CLEANING THE DATA
'''

'''
Step 1: Drops the first row, which contains the census variable IDs and the 
        two non-standard districts: Puerto Rico and Washington, DC then resets
        the index to: 0-434.
'''

df = df[df.fulldistrict != 'NAME']
df = df[df.fulldistrict != 'Delegate District (at Large) (113th Congress), District of Columbia']
df = df[df.fulldistrict != 'Resident Commissioner District (at Large) (113th Congress), Puerto Rico']

ind = range(0,435)
df['index'] = ind
df.set_index(df['index'], inplace=True)

'''
Step 2: Takes the 'fulldistrict' and splits it into 'district' and 'state' columns, 
        removing unecessary text.
'''

# Takes a string (the Census full name for the district), and returns list with
# two values: district number (int), full state name (str)

def GetDistrict(beak):
    district = beak.split('Congressional District ')
    district = district[1]
    district = district.split(' (113th Congress), ')
    if district[0] == '(at Large)':
        district[0] = 1
    else:
        district[0] = int(district[0])
    district[1] = str(district[1])
    return district

df['district'] = 0
df['statefull'] = ' '
i = 0
while i <= (len(df)-1):
    temp = GetDistrict(df.loc[i]['fulldistrict'])
    df['district'][i] = temp[0]
    df['statefull'][i] = temp[1]
    i += 1

''' Step 3: Clean up the table by dropping the unused columns: 'fulldistrict', 'state' '''

df.drop(['fulldistrict','state','index'], axis=1, inplace=True)

'''
SAVE THE DATA

This both backs up the data and converts many of the columns from strings to integers.

'''
df.to_csv('data/census_data.csv', index=False)
df = pd.read_csv('data/census_data.csv')
'''

ADDING NON CENSUS DATA FEATURES TO EACH CONGRESSIONAL DISTRICT

NOTE: Read census_data.csv back in to continue. This will convert all of the cells with numbers
into integers instead of strings as they were above.

Step 1: Adds in state abbreviations and creates a label column with the congressional district
        label: AL-4 for Alabama's fourt district, ND-1 for South Dakota's at-large district.

'''

''' Adding In The State Abbreviation And Congressional District Label '''

states = pd.read_csv('https://raw.githubusercontent.com/milleractual/congresshealth/master/data/states.csv')
df = pd.merge(df, states, on='statefull')

# Creates The District Labels

df['label'] = ' '
i = 0
while i <= (len(df)-1):
    df['label'][i] = '%s-%s' % (df['abbr'][i],df['district'][i])
    i += 1

''' 
Step 2: Add in the Cook PVI Ratings.  Ratings originated from a PDF produced in 2013.  The data was copied
        and formatted in Excel and saved as a CSV.
'''

cook = pd.read_csv('https://raw.githubusercontent.com/milleractual/congresshealth/master/data/pvi.csv')
cook.drop(['Sta','CD'], axis=1, inplace=True)

df = pd.merge(df, cook, on='label')

df['gopvi'] = 0
i = 0
while i <= (len(df)-1):
    if df.pvi[i][:1] == 'R':
        df.gopvi[i] = int(df.pvi[i][2:])
    else:
        df.gopvi[i] = 0 - int(df.pvi[i][2:])
    i += 1

'''
Step 3: Adds in the file on the actual size of the congressional district. Mileage matters.
'''

sqrm = pd.read_csv('https://raw.githubusercontent.com/milleractual/congresshealth/master/data/cdsqrmile.csv')
sqrm = pd.merge(sqrm, states, left_on='State', right_on='statefull')

sqrm['label'] = ' '
i = 0
while i <= (len(sqrm)-1):
    sqrm['label'][i] = '%s-%s' % (sqrm['abbr'][i],sqrm['District'][i])
    i += 1

sqrm.drop(['State','District','statefull','abbr'], axis=1, inplace=True)

df = pd.merge(df,sqrm, on='label')
df = df.rename(columns={'Land area (square miles)': 'sqrmiles'})

'''
Step 4: Create variables for the population density and the percentage of the total population that is white
'''

df['density'] = 0
i = 0
while i <= (len(df)-1):
    df.density[i] = df['population'][i] / df['sqrmiles'][i]
    i += 1

df['white'] = 0.0
i = 0
while i <= (len(df)-1):
    df.white[i] = float(df['popwhite'][i]) / float(df['population'][i])
    i += 1

'''
Step 5: Drop all the unused columns
'''

df.drop(['popwhite','popblack','pvi','Winner12','Margin12','Romney12','Obama12','Winner08','Margin08','McCain08','Obama08'],
         axis=1, inplace=True)

'''
SAVE THE DATA!
'''

df.to_csv('data/districtfeatures.csv', index=False)

'''
VARIABLES:

Feature         Type     Description
-------------------------------------------------------------------------------
population   :  int  :   Total population of the district

income       :  int  :   Median income of the district, in 2013 $US

homeval      :  int  :   Median home value in the district, in 2013 $US

rentval      :  int  :   Median rent cost in the district, in 2013 $US

district     :  int  :   Congressional District Number

statefull    :  str  :   Full name of the state

abbr         :  str  :   State Abbreviation

label        :  str  :   Congressional district ID, ie. Al-1 is Alabama's first congressional district

gopvi        :  int  :   GOP Voting Index - normalizes PVI to + for R leaning, and - for D leaning.

sqrmiles     :  flt  :   Total square miles for the district.

density      :  int  :   Number of people per square mile

white        :  flt  :   The percentage of the total population that is white 

-------------------------------------------------------------------------------

'''


