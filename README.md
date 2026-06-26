# Code IQ

<img width="1913" height="885" alt="image" src="https://github.com/user-attachments/assets/b3182167-3302-452c-9817-56c1390ea0c3" />

This is a AI-powered Code checker app which uses flask to operate.
It uses websockets and javascript to establish real-time code checking without refreshing the webpage
If you ever think the AI is making a mistake, you can send a query to the admins available shown on the admin panel.

# Cloning Instructions

<p>Type this in your terminal:</p>

```bash
git clone https://github.com/sarbarthasengupta2012-prog/HITL-Code-Checker-Website.git
cd HITL-Code-Checker-Website
```
<p>To run the server, enter this in your terminal: (This only works if you are in the HITL-Code-Checker-Website directory)</p>

```bash
python server.py
```
# How to navigate the website?

<ul>
  <li>Enter your code in the text box</li>
  <li>Click the button and check the result</li>
  <li>If you believe the results are wrong, then you can send the code to the admin panel for further checking.</li>
  <li>If the admin finds the AI made a mistake or didn't, then the code is sent to a MongoDB database where it is marked as 1 (clean) or 0 (messy) respectively.</li>
</ul>

# How does the AI work?
The AI is a logistic regression model. The data for the AI is stored inside a MongoDB and then transferred to a pkl file.
Whenever the MongoDB data is updated, make it a point to delete the pkl model. That way, it will recreate itself with the new data.

#### Note
This project is hosted on render! Check it out: https://hitl-code-checker-website.onrender.com
