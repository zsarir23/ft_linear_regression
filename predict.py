from linearRegression import LinearRegression

def main():
    lreg = LinearRegression()
    lreg.load_parameters()
    mileage = input('Enter mileage: ')
    try:
        mileage = float(mileage)
    except ValueError: 
        print('Please enter a valid number')
        return
    print(f'Estimated price for a car with mileage of {mileage} km is: {lreg.estimate_price(mileage)}')

if __name__ == '__main__':
    main()