product_price = [10,14,22,33,44,13,22,55,66,77]
totalSum = 0

print("Supermarket \n=========== ")

while True:
    userChoice = int(input("Please select the product number from 1 to 10 and 0 to Quit: "))
    if userChoice == 0:
        break
    elif 1<= userChoice<= 10:
        products = product_price[userChoice-1]
        totalSum += products
        print(f"Product: {userChoice} Price: {products}")
    else:
        print("Invalid product number.")

print(f"Total: {totalSum}")

paymentTaken = int(input("Payment: "))

changePayment = paymentTaken - totalSum

print(f"Change: {changePayment}")