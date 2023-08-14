"""
    crime-dataset-pipeline

       File: main.py

    Authors: Deleted for purposes of anonymity

    Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION

The software and its source code contain valuable trade secrets and shall be
maintained in confidence and treated as confidential information. The software
may only be used for evaluation and/or testing purposes, unless otherwise
explicitly stated in the terms of a license agreement or nondisclosure
agreement with the proprietor of the software. Any unauthorized publication,
transfer to third parties, or duplication of the object or source
code---either totally or in part---is strictly prohibited.

    Copyright (c) 2023 Proprietor: Deleted for purposes of anonymity
    All Rights Reserved.

THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.

NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
THE POSSIBILITY OF SUCH DAMAGES.

For purposes of anonymity, the identity of the proprietor is not given
herewith. The identity of the proprietor will be given once the review of the
conference submission is completed.

THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import stage


def main():
    # Logging
    stage.init_logging()

    # Prepare Database
    mdb = stage.init_database()

    # Load District Data
    stage.load_districts(mdb)

    # Load Crime Data
    crime_data_list = stage.load_crimes(mdb)

    # Load Street Data
    street_dataset = stage.load_streets(mdb, crime_data_list)

    # Load all Crime Documents from the Database
    crimes = stage.load_crime_database(mdb)

    # Load all Street Documents from the Database
    streets = stage.load_street_database(mdb)

    # Build links between Crime and Street Documents
    stage.link_crime_street(mdb, crimes, streets, street_dataset)

    # Search POIs
    poi_dataset = stage.search_pois(mdb, streets)

    # Load all POI Documents from the Database
    pois = stage.load_poi_database(mdb)

    # Build links between Street and POI Documents
    stage.link_street_poi(mdb, pois, poi_dataset)

    # Load all District Documents from the Database
    districts = stage.load_district_database(mdb)

    # Build links between District and Street/POI/Crime
    stage.link_district_all(mdb, districts, crimes, streets, pois)

    # Build links between Crimes and POI documents
    stage.link_crime_poi(mdb, crimes, pois)


if __name__ == '__main__':
    main()
