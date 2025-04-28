"""
Load testing script for the generative AI server.

- Simulates concurrent users sending text generation requests using Locust.
- Measures system behavior under load to check for slowdowns or failures.
"""

from locust import HttpUser, task, between


class AIUser(HttpUser):
    wait_time = between(1, 2)
    host = "http://localhost:8000"

    @task
    def gen(self):
        self.client.post("/generate", json={"prompt": "sunset over ocean"})
