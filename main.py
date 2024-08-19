import basic
while True:
    text = input("basic > ")
    result, error,context = basic.run('<stdin>',text)
    if error:print(error.as_string())
    elif context.types.iftype('NoPrint'):pass
    else:print(result)
