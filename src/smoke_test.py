import pandas as pd

def run_smoke_test():
    print("--- Running Mushroom Yield Smoke Test ---")
    
    # 1. Define the columns required by Task 1
    columns = ['temperature', 'humidity', 'CO2', 'yield']
    
    # 2. Add a sample dummy row representing your polyhouse sensors
    sample_row = [[24.5, 82.0, 650.0, 12.8]]
    
    # 3. Create a Pandas DataFrame
    df = pd.DataFrame(sample_row, columns=columns)
    
    # 4. Print it out beautifully
    print("\nSample Polyhouse Sensor Row:")
    print(df.to_string(index=False))
    print("\nSmoke test executed successfully!")

if __name__ == "__main__":
    run_smoke_test()