import pandas as pd

df = pd.read_csv("LU1319381_HKPolyU.txt", delim_whitespace=True, comment='#', skiprows=13)

print("Columns in DataFrame:")
print(df.columns.tolist())

df.columns = df.columns.str.strip() 
df = df.rename(columns={"Lonitude(degree)": "Longitude(degree)"}) 

df = df[(df["Diameter(m)"] > 0) & (df["Latitude(degree)"] != 0) & (df["Longitude(degree)"] != 0)]

df["Diameter_km"] = df["Diameter(m)"] / 1000.0

df.to_csv("cleaned_crater_catalog.csv", index=False)

print(f"Loaded {len(df)} craters with valid positions and diameters.")

