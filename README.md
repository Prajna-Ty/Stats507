This is a repo created for STATS 507 problem sets.

Here is the link to the solution of PS2Q3: [PS2Q3.ipynb](./PS2Q3.ipynb)

In the file, Pandas is used to read, clean, and append data files from the National Health and Nutrition Examination Survey (NHANES).
The data prepared here is a starting point for analyses in a few other problem sets.

For this problem, the four cohorts spanning the years 2011-2018 are used.

Here is a more detailed description of what each part of the PS2Q3.ipynb script does:

a) Use Python and Pandas to read and append the demographic datasets keeping only columns containing the unique ids (SEQN), age (RIDAGEYR), race and ethnicity (RIDRETH3), education (DMDEDUC2), and marital status (DMDMARTL), along with the following variables related to the survey weighting: (RIDSTATR, SDMVPSU, SDMVSTRA, WTMEC2YR, WTINT2YR). Add an additional column identifying to which cohort each case belongs. Rename the columns with literate variable names using all lower case and convert each column to an appropriate type. Finally, save the resulting data frame to a serialized “round-trip” format of your choosing (e.g. pickle, feather, or parquet).

b) Repeat part a for the oral health and dentition data (OHXDEN_*.XPT) retaining the following variables: SEQN, OHDDESTS, tooth counts (OHXxxTC), and coronal cavities (OHXxxCTC).

c) In your notebook, report the number of cases there are in the two datasets above.

Ref: https://jbhender.github.io/Stats507/F21/ps/ps2.html
