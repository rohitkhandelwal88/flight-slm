import pandas as pd

# Load the flight schedule data
df = pd.read_csv('data/flight_schedule.csv')

# Format it into structured text
df['formatted_text'] = df.apply(lambda row: f"Flight ID: {row['Flight ID']} | "
                                            f"Departure: {row['Departure Airport']} | "
                                            f"Arrival: {row['Arrival Airport']} | "
                                            f"Departure Time: {row['Departure Time']} | "
                                            f"Arrival Time: {row['Arrival Time']}", axis=1)

one_stop_flights = []
for _, flight1 in df.iterrows():
    for _, flight2 in df.iterrows():
        if flight1["Arrival Airport"] == flight2["Departure Airport"] and flight1["Departure Airport"] != flight2["Arrival Airport"]:
            one_stop_flights.append({
                "formatted_text": f"Flight ID: {flight1['Flight ID']} & {flight2['Flight ID']} | "
                                  f"Departure: {flight1['Departure Airport']} | "
                                  f"Stopover: {flight1['Arrival Airport']} | "
                                  f"Arrival: {flight2['Arrival Airport']} | "
                                  f"Departure Time: {flight1['Departure Time']} | "
                                  f"Stopover Time: {flight1['Arrival Time']} | "
                                  f"Arrival Time: {flight2['Arrival Time']} | "
                                  f"Type: One-Stop"
            })

# Convert to dataframe and merge
df_one_stop = pd.DataFrame(one_stop_flights)
df_combined = pd.concat([df[['formatted_text']], df_one_stop], ignore_index=True)
# Save formatted data
df_combined.to_csv('data/formatted_data.csv', index=False)
print("Data preprocessing complete. Saved to data/formatted_data.csv")
