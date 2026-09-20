# Download library to load a page, and query for HTML elements on that page.
import google.genai as genai
import asyncio
# from pyppeteer import launch
from playwright.async_api import async_playwright


import config

# Domino's Pizza
url = "https://www.google.com/maps/place/Domino's+Pizza/@37.6296762,-122.0482752,17z/data=!4m8!3m7!1s0x808f945afc883bdf:0x5d66c584c9f85cdc!8m2!3d37.6296762!4d-122.0482752!9m1!1b1!16s%2Fg%2F1tf_19jx?entry=ttu&g_ep=EgoyMDI2MDkxNi4wIKXMDSoASAFQAw%3D%3D"

# url = "https://www.google.com/maps/place/Domino's+Pizza/@37.6513166,-122.1225089,17z/data=!4m8!3m7!1s0x808f9122fd26852d:0xeb83d6e88f5c994b!8m2!3d37.6513124!4d-122.119934!9m1!1b1!16s%2Fg%2F1tff7vmc?entry=ttu&g_ep=EgoyMDI2MDkxNi4wIKXMDSoASAFQAw%3D%3D"

# Many pypeteer functions are async, so it will go perform a task and NOT wait for it to finish before moving on.
# Add await keyword so it knows to wait for the result, before moving on to next step.
# Wrap around in an async function.
async def scrape_reviews(url):
  reviews = []
  
  async with async_playwright() as p:
    
    # Create a browser. 
    # headless = False, lets us see the browser.
    # browser = await launch({"headless": True, "args": ["--window-size=800, 3200"]})
    browser = await p.chromium.launch(headless=False)


    # Set page size
    # page = await browser.newPage()
    # await page.setViewport({"width": 800, "height": 3200})
    page = await browser.new_page(viewport={"width": 800, "height": 3200})

    # Go to the URL.
    await page.goto(url)
    
    # Wait for all review divs to load on page
    await page.wait_for_selector('.jftiEf')
    elements = await page.query_selector_all('.jftiEf')
    
    for element in elements:
      try:
        # Find the more button so we can expand the review.
        # await page.wait_for_selector('.w8nwRe')
        more_btn = await element.query_selector('.w8nwRe')
        
        if more_btn is not None:
          # Click button if it exists.
          await more_btn.click()
          await page.wait_for_timeout(5000)
      except:
        pass
      
      try:
        # Get element for review
        # await page.waitForSelector('.MyEned')
        snippet = await element.query_selector('.MyEned')
        if snippet is not None:
          text = await snippet.text_content()
          reviews.append(text)
      except:
        pass
      
    # Close browser
    await browser.close()
    
    return reviews
  

# LLM Model to return summary.
def summarize(reviews, model):
  client = genai.Client(api_key=config.API_KEY)

  # Combine the reviews into a single block of text for the prompt
  reviews_text = "\n\n".join(f"Review {i+1}: {r}" for i, r in enumerate(reviews))

  prompt = f"""Below are customer reviews for a business. Summarize the overall
  sentiment, common themes (both positive and negative), and any recurring
  complaints or praise. Keep the summary concise, around 3-5 sentences.

  {reviews_text}
  """

  response = client.models.generate_content(
    model=model,
    contents=prompt, 
    # config=types.GenerateContentConfig(
    #   temperature=0.2
    # )
  )

  return response.text
  
reviews = asyncio.run(scrape_reviews(url))

summary = summarize(reviews, "gemini-3.6-flash")
print(summary)