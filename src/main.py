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
