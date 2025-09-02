items = []

while True:
    print("\n Would you like to\n 1. Add products to the list \n 2. Remove item from the list \n 3. Print the list and quit")
    userChoice = input("Your choice: ")

    if userChoice == "1":
        products = input("What will be added?: ")
        items.append(products)

    elif userChoice == "2":
        print(f"There are {len(items)} items in the list")
        try:
            idx = int(input("Which item is deleted?: "))
            items.pop(idx)
        except (ValueError, IndexError):
            print("Incorrect selection.")

    elif userChoice == "3":
        print("The following items remain in the list:")
        for i in items:
            print(i)
        break

    else:
        print("Incorrect selection.")
