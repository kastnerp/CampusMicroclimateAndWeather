# CampusMicroclimateAndWeather

# Gsheet Logging file

All notes and comments about battery status etc. [is collected here.](https://docs.google.com/spreadsheets/d/1gOeqPoUskun3UhIJ1qPwKxNo8b02dM2avpZNKrtnYA4/edit#gid=0)


# In general

- We use ReadPlot.ipynb for all the scripting
- Some helper functions live externally in the remaining *.py files

# Process:

1. Offload data from stations, in the .dtf format
2. export from hoboware in .csv, and make sure all the datetime settings are correct
3. open the csv, change the index so it starts at the next number we are adding to from the previous file (so not #1, #15,000 or whatever it is)
4. format the time correctly so that hoboreader python script can read it
5. Copy the new data into the master data file for that station (where we merge all the readouts), must be done in .xlsx file format so that excel does not mess up all the formatting
6. Save this new file as .csv in the github folder
7. Sync Github
8. re-run the python script, and almost always it complains about something, troubleshoot.

# Known Issues:

1. Hoboreader will not plot for the date unless all sources have data for that date. For instance, if 4 of your sources have data 1/1 - 6/1, and 1 has it 1/1 - 7/1, it will only plot until 6/1, right now.

2. The Pandas Apply function is nitpicky, and sometimes has to be re-run several times until it executes correctly. No error messages will show. 
