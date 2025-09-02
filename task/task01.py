def tester(givenstring = "Too short"):
    print(givenstring)

def main():
    while True:
        user_input = input("Write something (quit ends): ")
        if user_input == "quite":
            break
        if len(user_input) >= 10:
            tester(user_input)
        else:
            tester()
main()