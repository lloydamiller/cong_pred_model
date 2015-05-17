#Early Warning Indicators Of Campaigns

#### Background

Political campaigns all look the same when entering the summer of an election year.  Most have exited their primaries and are now fundraising and messaging for a general election against the opposing party.  Just as many are deep in the middle of party primaries/caucus/conventions.  However, there is a huge problem for those not privy to the internal details of every campaign: after July 15th, when the second quarter campaign finance reports are due, there is a blackout of information until October 15th when third quarter reports are due, just 2.5 weeks before the election.  In that time, major spending decisions need to be made by third party groups/independent expenditure committees, donors and fundraisers, professional operatives, and public figures interested in endorsing candidates (but only winners).

#### Project Question

How can we help predict the health of a campaign, and its likeliness to succeed on that first Tuesday in November, from the data we have on July 15th?

#### The Data

I will be looking at data that has both voter characteristics (district data like population, median income, etc) that are generally used to examine voter behavior and campaign dynamics, as well as political characteristics (partisan index, past election results).  This will result in approximately 42 features for each of the 870 instances of a congressional election in 2012 and 2014 (435 districts). 

##### Static District Data
Source: The U.S. Census Bureau American Community Survey 3-Year, API; Cook does not publish full scores for all districts over time, but I have obtained their 2012 scores which I will use.
- District population
- Population by Race (white, black, other)
- Median Income
- Median House Cost
- Median Rent Cost
- % of population with high school diploma
- % of population with bachelors degree
- Cook Political Report Partisan Voter Index Score (1)


##### Campaign Finance Data
Source: Federal Election Commission data, each item will be broken into quarters (By Quarter = Q1-4 off year, Q1-2 on year  = 6 segments)
- Total Raised
- Total Loaned To Campaign + Total Contributions by Candidate (self funding score)
- Total Spent by Campaign
- % of Total Raised from PAC/Committees
- % of Total Spent on payroll
- % of Total Spent on fundraising


##### Identify General Election Campaign Committees + Obtain Election Results
To determine which campaigns those are, I will have to go by hand through election results and match candidates to their committees.  This will create a data table with the following information for each 870 races:
- R Candidate Raw Votes
- D Candidates Raw Votes
This data is the “response” that I am trying to predict.

Note that, at this time, I have not excluded non-competitive races where there was no general election candidate.