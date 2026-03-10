package models

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOverrideProvider(t *testing.T) {
	catalog := &Catalog{
		Providers: map[string][]ModelPricing{
			"test": {
				{ID: "model-a", InputPricePer1M: 1.0, OutputPricePer1M: 2.0, ContextWindow: 1000, MaxOutput: 500},
				{ID: "model-b", InputPricePer1M: 3.0, OutputPricePer1M: 6.0, ContextWindow: 2000, MaxOutput: 1000},
			},
		},
	}

	overrides := []ModelPricing{
		// Override pricing for model-a, leave context/output unchanged
		{ID: "model-a", InputPricePer1M: 1.5, OutputPricePer1M: 2.5},
		// Override context for model-b only
		{ID: "model-b", ContextWindow: 4000},
		// New model not in catalog
		{ID: "model-c", InputPricePer1M: 5.0, OutputPricePer1M: 10.0, ContextWindow: 8000, MaxOutput: 2000},
	}

	overrideProvider(catalog, "test", overrides)

	models := catalog.ForProvider("test")
	if len(models) != 3 {
		t.Fatalf("got %d models, want 3", len(models))
	}

	// Find each model
	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	// model-a: pricing overridden, context/output unchanged
	a := byID["model-a"]
	if a.InputPricePer1M != 1.5 {
		t.Errorf("model-a input = %f, want 1.5", a.InputPricePer1M)
	}
	if a.OutputPricePer1M != 2.5 {
		t.Errorf("model-a output = %f, want 2.5", a.OutputPricePer1M)
	}
	if a.ContextWindow != 1000 {
		t.Errorf("model-a context = %d, want 1000 (unchanged)", a.ContextWindow)
	}
	if a.MaxOutput != 500 {
		t.Errorf("model-a maxOutput = %d, want 500 (unchanged)", a.MaxOutput)
	}

	// model-b: only context overridden
	b := byID["model-b"]
	if b.InputPricePer1M != 3.0 {
		t.Errorf("model-b input = %f, want 3.0 (unchanged)", b.InputPricePer1M)
	}
	if b.ContextWindow != 4000 {
		t.Errorf("model-b context = %d, want 4000", b.ContextWindow)
	}

	// model-c: added new
	c := byID["model-c"]
	if c.InputPricePer1M != 5.0 {
		t.Errorf("model-c input = %f, want 5.0", c.InputPricePer1M)
	}
	if c.ContextWindow != 8000 {
		t.Errorf("model-c context = %d, want 8000", c.ContextWindow)
	}
}

func TestParseOpenAIText(t *testing.T) {
	text := `# GPT-4o

**Current Snapshot:** gpt-4o-2024-08-06

## Snapshots

### gpt-4o-2024-08-06

- Context window size: 128000
- Knowledge cutoff date: 2023-10-01
- Maximum output tokens: 16384

### gpt-4o-mini-2024-07-18

- Context window size: 128000
- Maximum output tokens: 16384

## Text tokens

| Name | Input | Cached input | Output | Unit |
| --- | --- | --- | --- | --- |
| gpt-4o | 2.5 | 1.25 | 10 | 1M tokens |
| gpt-4o (batch) | 1.25 | | 5 | 1M tokens |
| gpt-4o-2024-08-06 | 2.5 | 1.25 | 10 | 1M tokens |
| gpt-4o-mini | 0.15 | 0.075 | 0.6 | 1M tokens |
| o1-pro | 100 | | 400 | 1M tokens |

## Audio tokens
`

	models, err := parseOpenAIText(text)
	if err != nil {
		t.Fatalf("parseOpenAIText error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	// gpt-4o: pricing only (no metadata section for family name)
	if m, ok := byID["gpt-4o"]; !ok {
		t.Error("gpt-4o not found")
	} else {
		if m.InputPricePer1M != 2.5 {
			t.Errorf("gpt-4o input = %f, want 2.5", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 10 {
			t.Errorf("gpt-4o output = %f, want 10", m.OutputPricePer1M)
		}
	}

	// gpt-4o-2024-08-06: pricing + metadata
	if m, ok := byID["gpt-4o-2024-08-06"]; !ok {
		t.Error("gpt-4o-2024-08-06 not found")
	} else {
		if m.InputPricePer1M != 2.5 {
			t.Errorf("gpt-4o-2024-08-06 input = %f, want 2.5", m.InputPricePer1M)
		}
		if m.ContextWindow != 128000 {
			t.Errorf("gpt-4o-2024-08-06 context = %d, want 128000", m.ContextWindow)
		}
		if m.MaxOutput != 16384 {
			t.Errorf("gpt-4o-2024-08-06 maxOutput = %d, want 16384", m.MaxOutput)
		}
	}

	// gpt-4o-mini-2024-07-18: metadata only (no pricing row for this snapshot)
	if m, ok := byID["gpt-4o-mini-2024-07-18"]; !ok {
		t.Error("gpt-4o-mini-2024-07-18 not found")
	} else {
		if m.ContextWindow != 128000 {
			t.Errorf("context = %d, want 128000", m.ContextWindow)
		}
		if m.InputPricePer1M != 0 {
			t.Errorf("input = %f, want 0 (no pricing row)", m.InputPricePer1M)
		}
	}

	// Batch entries should be filtered
	if _, ok := byID["gpt-4o (batch)"]; ok {
		t.Error("batch entry should be filtered out")
	}

	// o1-pro: high pricing
	if m, ok := byID["o1-pro"]; !ok {
		t.Error("o1-pro not found")
	} else {
		if m.InputPricePer1M != 100 {
			t.Errorf("o1-pro input = %f, want 100", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 400 {
			t.Errorf("o1-pro output = %f, want 400", m.OutputPricePer1M)
		}
	}
}

func TestParseAnthropicHTML(t *testing.T) {
	html := `<html><body>
<table>
<tr><th>Feature</th><th>Claude Opus 4.6</th><th>Claude Sonnet 4.6</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-opus-4-6</td><td>claude-sonnet-4-6</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-opus-4-6</td><td>claude-sonnet-4-6</td></tr>
<tr><td><strong>Pricing</strong></td><td>$5 / input MTok<br/>$25 / output MTok</td><td>$3 / input MTok<br/>$15 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>200K tokens</td><td>200K tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>128K tokens</td><td>64K tokens</td></tr>
</table>
<table>
<tr><th>Feature</th><th>Claude Sonnet 4.5</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-sonnet-4-5-20250929</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-sonnet-4-5</td></tr>
<tr><td><strong>Pricing</strong></td><td>$3 / input MTok<br/>$15 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>200K tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>64K tokens</td></tr>
</table>
</body></html>`

	models, err := parseAnthropicHTML(html)
	if err != nil {
		t.Fatalf("parseAnthropicHTML error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	// Opus 4.6
	if m, ok := byID["claude-opus-4-6"]; !ok {
		t.Error("claude-opus-4-6 not found")
	} else {
		if m.InputPricePer1M != 5 {
			t.Errorf("opus input = %f, want 5", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 25 {
			t.Errorf("opus output = %f, want 25", m.OutputPricePer1M)
		}
		if m.ContextWindow != 200000 {
			t.Errorf("opus context = %d, want 200000", m.ContextWindow)
		}
		if m.MaxOutput != 128000 {
			t.Errorf("opus maxOutput = %d, want 128000", m.MaxOutput)
		}
	}

	// Sonnet 4.6
	if m, ok := byID["claude-sonnet-4-6"]; !ok {
		t.Error("claude-sonnet-4-6 not found")
	} else {
		if m.OutputPricePer1M != 15 {
			t.Errorf("sonnet output = %f, want 15", m.OutputPricePer1M)
		}
		if m.MaxOutput != 64000 {
			t.Errorf("sonnet maxOutput = %d, want 64000", m.MaxOutput)
		}
	}

	// Sonnet 4.5 (legacy) — both ID and alias should be present
	if _, ok := byID["claude-sonnet-4-5-20250929"]; !ok {
		t.Error("claude-sonnet-4-5-20250929 not found")
	}
	if m, ok := byID["claude-sonnet-4-5"]; !ok {
		t.Error("claude-sonnet-4-5 alias not found")
	} else {
		if m.InputPricePer1M != 3 {
			t.Errorf("sonnet 4.5 alias input = %f, want 3", m.InputPricePer1M)
		}
	}
}

func TestParseGrokPage(t *testing.T) {
	// Simulates the RSC payload format found in the xAI docs page
	html := `<html><script>
some stuff \"name\":\"grok-3\",\"version\":\"1.0\",\"inputModalities\":[1],\"outputModalities\":[1],\"promptTextTokenPrice\":\"$n30000\",\"completionTextTokenPrice\":\"$n150000\",\"maxPromptLength\":131072
more stuff \"name\":\"grok-3-mini\",\"version\":\"1.0\",\"promptTextTokenPrice\":\"$n3000\",\"completionTextTokenPrice\":\"$n5000\",\"maxPromptLength\":131072
\"name\":\"grok-4-fast-reasoning\",\"version\":\"1.0\",\"promptTextTokenPrice\":\"$n2000\",\"completionTextTokenPrice\":\"$n5000\",\"maxPromptLength\":2000000
\"name\":\"grok-imagine-image\",\"version\":\"1.0\",\"promptTextTokenPrice\":\"$n0\",\"completionTextTokenPrice\":\"$n0\",\"maxPromptLength\":0
duplicate \"name\":\"grok-3\",\"version\":\"1.0\",\"promptTextTokenPrice\":\"$n30000\",\"completionTextTokenPrice\":\"$n150000\",\"maxPromptLength\":131072
</script></html>`

	models, err := parseGrokPage(html)
	if err != nil {
		t.Fatalf("parseGrokPage error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	// grok-3: $n30000/10000 = $3.00, $n150000/10000 = $15.00
	if m, ok := byID["grok-3"]; !ok {
		t.Error("grok-3 not found")
	} else {
		if m.InputPricePer1M != 3.0 {
			t.Errorf("grok-3 input = %f, want 3.0", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 15.0 {
			t.Errorf("grok-3 output = %f, want 15.0", m.OutputPricePer1M)
		}
		if m.ContextWindow != 131072 {
			t.Errorf("grok-3 context = %d, want 131072", m.ContextWindow)
		}
	}

	// grok-4-fast-reasoning: $n2000/10000 = $0.20
	if m, ok := byID["grok-4-fast-reasoning"]; !ok {
		t.Error("grok-4-fast-reasoning not found")
	} else {
		if m.InputPricePer1M != 0.2 {
			t.Errorf("input = %f, want 0.2", m.InputPricePer1M)
		}
		if m.ContextWindow != 2000000 {
			t.Errorf("context = %d, want 2000000", m.ContextWindow)
		}
	}

	// grok-imagine-image should be filtered out
	if _, ok := byID["grok-imagine-image"]; ok {
		t.Error("image model should be filtered out")
	}

	// Duplicates should be deduplicated
	if len(models) != 3 {
		t.Errorf("got %d models, want 3 (deduplicated)", len(models))
	}
}

func TestParseGeminiHTML(t *testing.T) {
	// Simulate stripped HTML text content
	html := `<html><body>
<h3>Gemini 2.5 Pro</h3>
<p>Input price $1.25</p>
<p>Output price $10.00</p>

<h3>Gemini 2.5 Flash</h3>
<p>Input price $0.30 (text / image / video)</p>
<p>Output price (including thinking tokens) $2.50</p>

<h3>Gemini 2.5 Flash-Lite</h3>
<p>Input price $0.10</p>
<p>Output price $0.40</p>

<h3>Gemini 2.5 Flash Image</h3>
<p>Output price $30.00</p>
</body></html>`

	models, err := parseGeminiHTML(html)
	if err != nil {
		t.Fatalf("parseGeminiHTML error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	if m, ok := byID["gemini-2.5-pro"]; !ok {
		t.Error("gemini-2.5-pro not found")
	} else {
		if m.InputPricePer1M != 1.25 {
			t.Errorf("input = %f, want 1.25", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 10.0 {
			t.Errorf("output = %f, want 10.0", m.OutputPricePer1M)
		}
	}

	if m, ok := byID["gemini-2.5-flash"]; !ok {
		t.Error("gemini-2.5-flash not found")
	} else {
		if m.InputPricePer1M != 0.30 {
			t.Errorf("input = %f, want 0.30", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 2.50 {
			t.Errorf("output = %f, want 2.50", m.OutputPricePer1M)
		}
	}

	if m, ok := byID["gemini-2.5-flash-lite"]; !ok {
		t.Error("gemini-2.5-flash-lite not found")
	} else {
		if m.InputPricePer1M != 0.10 {
			t.Errorf("input = %f, want 0.10", m.InputPricePer1M)
		}
	}

	// Image model should be skipped
	if _, ok := byID["gemini-2.5-flash-image"]; ok {
		t.Error("image model should be filtered")
	}
}

func TestEnrichFromProviders_AllSucceed(t *testing.T) {
	openAIServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`## Text tokens

| Name | Input | Cached input | Output | Unit |
| --- | --- | --- | --- | --- |
| gpt-4o | 2.5 | 1.25 | 10 | 1M tokens |
`))
	}))
	defer openAIServer.Close()

	anthropicServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<table>
<tr><th>Feature</th><th>Claude Opus 4.6</th></tr>
<tr><td>Claude API ID</td><td>claude-opus-4-6</td></tr>
<tr><td>Pricing</td><td>$5 / input MTok $25 / output MTok</td></tr>
<tr><td>Context window</td><td>200K tokens</td></tr>
<tr><td>Max output</td><td>128K tokens</td></tr>
</table>`))
	}))
	defer anthropicServer.Close()

	geminiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<h3>Gemini 2.5 Flash</h3><p>Input price $0.30</p><p>Output price $2.50</p>`))
	}))
	defer geminiServer.Close()

	grokServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`\"name\":\"grok-3\",\"promptTextTokenPrice\":\"$n30000\",\"completionTextTokenPrice\":\"$n150000\",\"maxPromptLength\":131072`))
	}))
	defer grokServer.Close()

	origOpenAI, origAnthropic, origGemini, origGrok := openAISourceURL, anthropicSourceURL, geminiSourceURL, grokSourceURL
	defer func() {
		openAISourceURL = origOpenAI
		anthropicSourceURL = origAnthropic
		geminiSourceURL = origGemini
		grokSourceURL = origGrok
	}()
	openAISourceURL = openAIServer.URL
	anthropicSourceURL = anthropicServer.URL
	geminiSourceURL = geminiServer.URL
	grokSourceURL = grokServer.URL

	catalog := &Catalog{
		Source: "litellm",
		Providers: map[string][]ModelPricing{
			"openai": {
				{ID: "gpt-4o", InputPricePer1M: 2.0, OutputPricePer1M: 8.0, ContextWindow: 128000, MaxOutput: 16384},
			},
			"grok": {
				{ID: "grok-3", InputPricePer1M: 0, OutputPricePer1M: 0, ContextWindow: 0, MaxOutput: 0},
			},
		},
	}

	if err := EnrichFromProviders(context.Background(), catalog); err != nil {
		t.Fatalf("EnrichFromProviders() error: %v", err)
	}

	// OpenAI should be overridden
	for _, m := range catalog.ForProvider("openai") {
		if m.ID == "gpt-4o" {
			if m.InputPricePer1M != 2.5 {
				t.Errorf("gpt-4o input = %f, want 2.5 (overridden)", m.InputPricePer1M)
			}
			if m.OutputPricePer1M != 10 {
				t.Errorf("gpt-4o output = %f, want 10 (overridden)", m.OutputPricePer1M)
			}
			if m.ContextWindow != 128000 {
				t.Errorf("gpt-4o context = %d, want 128000 (unchanged)", m.ContextWindow)
			}
		}
	}

	// Grok should be enriched
	for _, m := range catalog.ForProvider("grok") {
		if m.ID == "grok-3" {
			if m.InputPricePer1M != 3.0 {
				t.Errorf("grok-3 input = %f, want 3.0", m.InputPricePer1M)
			}
			if m.ContextWindow != 131072 {
				t.Errorf("grok-3 context = %d, want 131072", m.ContextWindow)
			}
		}
	}
}

func TestEnrichFromProviders_PartialFailure(t *testing.T) {
	openAIServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`## Text tokens

| Name | Input | Cached input | Output | Unit |
| --- | --- | --- | --- | --- |
| gpt-4o | 2.5 | 1.25 | 10 | 1M tokens |
`))
	}))
	defer openAIServer.Close()

	errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer errorServer.Close()

	origOpenAI, origAnthropic, origGemini, origGrok := openAISourceURL, anthropicSourceURL, geminiSourceURL, grokSourceURL
	defer func() {
		openAISourceURL = origOpenAI
		anthropicSourceURL = origAnthropic
		geminiSourceURL = origGemini
		grokSourceURL = origGrok
	}()
	openAISourceURL = openAIServer.URL
	anthropicSourceURL = errorServer.URL // fails
	geminiSourceURL = errorServer.URL    // fails
	grokSourceURL = errorServer.URL      // fails

	catalog := &Catalog{
		Source: "litellm",
		Providers: map[string][]ModelPricing{
			"openai": {
				{ID: "gpt-4o", InputPricePer1M: 2.0, OutputPricePer1M: 8.0},
			},
		},
	}

	err := EnrichFromProviders(context.Background(), catalog)
	if err == nil {
		t.Fatal("expected error when providers fail")
	}

	// Catalog should be unchanged (no overrides applied)
	for _, m := range catalog.ForProvider("openai") {
		if m.ID == "gpt-4o" && m.InputPricePer1M != 2.0 {
			t.Errorf("gpt-4o input = %f, want 2.0 (should be unchanged on failure)", m.InputPricePer1M)
		}
	}
}

func TestEnrichFromProviders_NilCatalog(t *testing.T) {
	// Should not panic or error
	if err := EnrichFromProviders(context.Background(), nil); err != nil {
		t.Errorf("unexpected error for nil catalog: %v", err)
	}

	catalog := &Catalog{}
	if err := EnrichFromProviders(context.Background(), catalog); err != nil {
		t.Errorf("unexpected error for empty catalog: %v", err)
	}
}

func TestParseHelpers(t *testing.T) {
	t.Run("parseTokenCount", func(t *testing.T) {
		tests := []struct {
			input string
			want  int
		}{
			{"200K tokens", 200000},
			{"1M tokens", 1000000},
			{"128K tokens", 128000},
			{"64K tokens", 64000},
			{"no match", 0},
		}
		for _, tt := range tests {
			if got := parseTokenCount(tt.input); got != tt.want {
				t.Errorf("parseTokenCount(%q) = %d, want %d", tt.input, got, tt.want)
			}
		}
	})

	t.Run("parseSizeStr", func(t *testing.T) {
		tests := []struct {
			input string
			want  int
		}{
			{"2M", 2000000},
			{"256K", 256000},
			{"131K", 131000},
			{"", 0},
		}
		for _, tt := range tests {
			if got := parseSizeStr(tt.input); got != tt.want {
				t.Errorf("parseSizeStr(%q) = %d, want %d", tt.input, got, tt.want)
			}
		}
	})

	t.Run("parseDollarAmount", func(t *testing.T) {
		tests := []struct {
			input string
			want  float64
		}{
			{"$3.00", 3.0},
			{"$0.20", 0.2},
			{"$15.00", 15.0},
			{"no price", 0},
		}
		for _, tt := range tests {
			if got := parseDollarAmount(tt.input); got != tt.want {
				t.Errorf("parseDollarAmount(%q) = %f, want %f", tt.input, got, tt.want)
			}
		}
	})

	t.Run("stripHTMLTags", func(t *testing.T) {
		input := `<strong>Bold</strong> and <a href="x">link</a>`
		got := stripHTMLTags(input)
		if got != "Bold and link" {
			t.Errorf("stripHTMLTags = %q, want %q", got, "Bold and link")
		}
	})
}
