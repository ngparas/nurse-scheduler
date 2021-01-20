# Nurse Scheduler

## Overview
This project provides an easy, calendar interface for a Lead Nurse to create the schedule for their team. The team is outpatient and is responsible for covering several different types of clinics and on-call rotations.

The user should be able to fill out:

* their team members & which clinics they are qualified to cover
* the clinic dates
* their team members' availability
* other responsibilities that need to be covered (e.g. on-call, charge)

Then the application should produce an optimized schedule. Additionally, that schedule should satisfy the following constraints:

* The same person cannot be on-call two days in a row
* Some nurses cannot cover certain clinic types (due to language requirements, etc.)
* A nurse cannot be on-call and charge at the same time
* A nurse cannot be in more than 1 clinic at a time
* A nurse cannot be on-call and in a clinic at the same time
* The schedule must respect availability (e.g. PTO)
* A nurse cannot be in clinic/charge the day after being on-call
* The nurse on-call on saturday should have friday clear. The Nurse on-call on sunday should have monday clear.

Lastly, the objective function should penalize certain undesirable outcomes such as: 

* "Sandwich Call", where someone is on-call twice with only 1 day in-between
* Should try to respect "no call" requests

The schedule should work in units of full days, ie a nurse should be scheduled in a clinic for a full day at a time. The schedule should be made 1 month at a time.

The app should also support "fixing" certain assignments. For example, a given nurse may have been assigned a holiday months in advance. That should be pre-filled before the optimizer runs, and remain fixed throughout.
