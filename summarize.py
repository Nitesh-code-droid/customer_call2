import cohere
co = cohere.Client('ynnI8KhI8tUOTkpjdluAfi265Yz2YpkVSWXWLwXh')
def from_cohere(text, emotion):
    custom_query = f"for this text : '{text}'. I got '{emotion} list with their respective probabilities' as result. so I need you to summerize the '{text}' how this text is '{emotion}'. please make the response be 3 lines short and precise."
    response = co.generate(
    prompt=custom_query,
    ) 
    print(response)
    return response[0].text