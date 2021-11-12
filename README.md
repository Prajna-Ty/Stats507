This is a repo created for STATS 507 problem sets.

## PS2Q3

Here is the link to the solution of PS2Q3: [PS2Q3.ipynb](./PS2Q3.ipynb)

In the file, Pandas is used to read, clean, and append data files from the National Health and Nutrition Examination Survey (NHANES).
The data prepared here is a starting point for analyses in a few other problem sets.

For this problem, the four cohorts spanning the years 2011-2018 are used.

Here is a more detailed description of what each part of the PS2Q3.ipynb script does:

a)
- Use Python and Pandas to read and append the demographic datasets keeping only columns containing the unique ids (SEQN), age (RIDAGEYR), race and ethnicity (RIDRETH3), education (DMDEDUC2), and marital status (DMDMARTL), along with the following variables related to the survey weighting: (RIDSTATR, SDMVPSU, SDMVSTRA, WTMEC2YR, WTINT2YR).

- Add an additional column identifying to which cohort each case belongs.

- Rename the columns with literate variable names using all lower case and convert each column to an appropriate type.

- Finally, save the resulting data frame to pickle format.

b) Repeat part a for the oral health and dentition data (OHXDEN_*.XPT) retaining the following variables: SEQN, OHDDESTS, tooth counts (OHXxxTC), and coronal cavities (OHXxxCTC).

c) Report the number of cases in the two datasets above.

Ref: https://jbhender.github.io/Stats507/F21/ps/ps2.html

## PS4Q0 - Topics in Pandas

Here is the link to the solution of PS4Q0: [PS4Q0.ipynb](./PS4Q0.ipynb)

Requirement for PS4Q0: Pick a topic - such as a function, class, method,
recipe or idiom related to the pandas python library
and create a short tutorial or overview of that topic. 

1. Pick a topic not covered in the class slides.

2. Use bullet points and titles (level 2 headers)
to create the equivalent of 3-5 “slides” of key points.
They shouldn’t actually be slides,
but please structure your key points
in a manner similar to the class slides
(viewed as a notebook).

3. Include executable example code in code cells
to illustrate your topic.

Ref: https://jbhender.github.io/Stats507/F21/ps/ps4.html

## Link to commit history
https://github.com/Prajna-Ty/Stats507/commits/main
