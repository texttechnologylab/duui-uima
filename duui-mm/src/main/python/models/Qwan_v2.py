class Qwen2_5VL:
    def __init__(self,
                 api_url="http://localhost:6659/v1/chat/completions",
                 model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                 model_version="latest",
                 model_lang="multi",
                 model_source="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
                 logging_level="INFO"):

        self.api_url = api_url
        self.model_name = model_name
        self.model_version = model_version
        self.model_lang = model_lang
        self.model_source = model_source

        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

    def _generate_dummy_ref(self):
        return str(uuid4().int % 1_000_000)

    def _build_chat_request(self, messages):
        return {
            "model": self.model_name,
            "messages": messages,
        }

    def _check_sleep_state(self):
        try:
            res = requests.get(f"{self.api_url.replace('/v1/chat/completions','')}/is_sleep")
            return res.json().get("is_sleep", False)
        except Exception:
            return False

    def _wake_up_and_sleep_other(self, other_url):
        requests.post(f"{self.api_url.replace('/v1/chat/completions','')}/wake_up")
        requests.post(f"{other_url}/sleep")

    @handle_errors
    def process_text(self, prompt: LLMPrompt, other_url: str) -> LLMResult:
        if self._check_sleep_state():
            self._wake_up_and_sleep_other(other_url)

        messages = [{"role": m.role, "content": m.content} for m in prompt.messages]
        body = self._build_chat_request(messages)
        headers = {"Content-Type": "application/json"}

        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()
        generated_text = result["choices"][0]["message"]["content"]
        return LLMResult(
            meta=json.dumps({"response": generated_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    # Add process_image, process_audio, etc., following similar pattern
