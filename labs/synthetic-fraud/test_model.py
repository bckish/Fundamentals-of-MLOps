import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from model import score, train
from serve import make_handler


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train()

    def test_reproducibility_and_evaluation(self):
        self.assertEqual(self.model, train())
        self.assertGreater(self.model["metrics"]["accuracy"], 0.85)
        self.assertEqual(self.model["metrics"]["heldout_rows"], 160)
        self.assertGreater(score(self.model, [0.9, 0.9, 0.9]), score(self.model, [0.1, 0.1, 0.1]))

    def test_reject_invalid_inputs(self):
        for value in [[], [True, 0, 0], [float("nan"), 0, 0], [2, 0, 0], [10**400, 0, 0]]:
            with self.assertRaises(ValueError):
                score(self.model, value)

    def test_http_and_telemetry(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.model))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/healthz") as response:
                self.assertEqual(json.load(response)["status"], "ok")
            with urlopen(Request(base + "/predict", data=b'{"features":[0.9,0.9,0.9]}', headers={"Content-Type":"application/json"})) as response:
                self.assertTrue(json.load(response)["synthetic"])
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + "/predict", data=b'{"features":[]}'))
            self.assertEqual(error.exception.code, 400)
            with urlopen(base + "/metrics") as response:
                metrics = response.read().decode()
                self.assertIn("fraud_predictions_total 1", metrics)
                self.assertIn("fraud_invalid_requests_total 1", metrics)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
