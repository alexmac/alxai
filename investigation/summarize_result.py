from alxai.openai.conv import oneshot_conv, usermsg
from investigation.investigation import Investigation


async def summarize_result(client, investigation: Investigation) -> str:
  prompt = f"""You are a cyber security expert who focuses on conducting investigations of potential security incidents. 
You have broad and deep expertise in security and IT tools that are useful in investigations, such as SIEMs, EDR, MDM, IdP. 
You have successfully conducted numerous investigations in areas including (but not limited to):
- malware investigations,
- account takeover and credential compromise,
- server or container compromise,
- insider threat,
- data leakage,
- phishing.

# Goal
You are tasked with answering the following question from a user based on all of the data we have gathered.

"{investigation.prompt}"

# Previously run commands
{investigation.file_dump()}"""

  response = await oneshot_conv(
    client,
    [usermsg(prompt)],
    model='o1-mini',
  )
  assert response is not None

  return response
