package models

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Source URLs for official provider data (variables for testability).
var (
	openAISourceURL    = "https://cdn.openai.com/API/docs/txt/llms-models-pricing.txt"
	anthropicSourceURL = "https://platform.claude.com/docs/en/docs/about-claude/models"
	geminiSourceURL    = "https://ai.google.dev/pricing"
	grokSourceURL      = "https://docs.x.ai/docs/models"
)

// EnrichFromProviders fetches model data from official provider sources
// and overrides catalog entries where official data is available.
// All provider fetches must succeed; if any fails, the catalog is left
// unchanged and an error is returned.
func EnrichFromProviders(ctx context.Context, catalog *Catalog) error {
	if catalog == nil || catalog.Providers == nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	type result struct {
		provider string
		models   []ModelPricing
		err      error
	}

	type fetcher struct {
		provider string
		fn       func(context.Context) ([]ModelPricing, error)
	}

	fetchers := []fetcher{
		{"openai", fetchOpenAIModels},
		{"anthropic", fetchAnthropicModels},
		{"gemini", fetchGeminiModels},
		{"grok", fetchGrokModels},
	}

	results := make([]result, len(fetchers))
	var wg sync.WaitGroup
	for i, f := range fetchers {
		wg.Add(1)
		go func(i int, f fetcher) {
			defer wg.Done()
			models, err := f.fn(ctx)
			results[i] = result{provider: f.provider, models: models, err: err}
		}(i, f)
	}
	wg.Wait()

	// If any provider failed, return error without modifying the catalog
	var errs []string
	for _, r := range results {
		if r.err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", r.provider, r.err))
		} else if len(r.models) == 0 {
			errs = append(errs, fmt.Sprintf("%s: no models found", r.provider))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("provider enrichment failed: %s", strings.Join(errs, "; "))
	}

	// All succeeded — apply overrides
	for _, r := range results {
		overrideProvider(catalog, r.provider, r.models)
	}
	return nil
}

// overrideProvider merges official data into the catalog for a provider.
// Non-zero fields from overrides replace existing values.
// Models not in the catalog are added.
func overrideProvider(catalog *Catalog, provider string, overrides []ModelPricing) {
	existing := catalog.Providers[provider]

	overrideMap := make(map[string]ModelPricing, len(overrides))
	for _, m := range overrides {
		overrideMap[m.ID] = m
	}

	for i, m := range existing {
		o, ok := overrideMap[m.ID]
		if !ok {
			continue
		}
		if o.InputPricePer1M > 0 {
			existing[i].InputPricePer1M = o.InputPricePer1M
		}
		if o.OutputPricePer1M > 0 {
			existing[i].OutputPricePer1M = o.OutputPricePer1M
		}
		if o.ContextWindow > 0 {
			existing[i].ContextWindow = o.ContextWindow
		}
		if o.MaxOutput > 0 {
			existing[i].MaxOutput = o.MaxOutput
		}
		delete(overrideMap, m.ID)
	}

	for _, m := range overrideMap {
		if m.ID != "" {
			existing = append(existing, m)
		}
	}
	catalog.Providers[provider] = existing
}

// --- Shared helpers ---

func providerHTTPGet(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "langdag-model-catalog/1.0")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

var (
	htmlTagRe    = regexp.MustCompile(`<[^>]*>`)
	whitespaceRe = regexp.MustCompile(`\s+`)
	tableRe      = regexp.MustCompile(`(?s)<table[^>]*>(.*?)</table>`)
	rowRe        = regexp.MustCompile(`(?s)<tr[^>]*>(.*?)</tr>`)
	cellRe       = regexp.MustCompile(`(?s)<t[dh][^>]*>(.*?)</t[dh]>`)
	tokenCountRe = regexp.MustCompile(`([\d.]+)\s*([KkMm])\s*tokens`)
	sizeRe       = regexp.MustCompile(`([\d.]+)\s*([KkMm])`)
	dollarRe     = regexp.MustCompile(`\$\s*([\d.]+)`)
)

func stripHTMLTags(s string) string {
	s = htmlTagRe.ReplaceAllString(s, " ")
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", `"`)
	s = strings.ReplaceAll(s, "&#39;", "'")
	s = strings.ReplaceAll(s, "&nbsp;", " ")
	s = strings.ReplaceAll(s, `\$`, "$")
	s = whitespaceRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

// parseHTMLTables extracts tables from HTML as [table][row][cell] strings.
func parseHTMLTables(html string) [][][]string {
	var tables [][][]string
	for _, tm := range tableRe.FindAllStringSubmatch(html, -1) {
		var rows [][]string
		for _, rm := range rowRe.FindAllStringSubmatch(tm[1], -1) {
			var cells []string
			for _, cm := range cellRe.FindAllStringSubmatch(rm[1], -1) {
				cells = append(cells, stripHTMLTags(cm[1]))
			}
			if len(cells) > 0 {
				rows = append(rows, cells)
			}
		}
		if len(rows) > 0 {
			tables = append(tables, rows)
		}
	}
	return tables
}

func parseTokenCount(s string) int {
	m := tokenCountRe.FindStringSubmatch(s)
	if m == nil {
		return 0
	}
	return parseSizeValue(m[1], m[2])
}

func parseSizeStr(s string) int {
	m := sizeRe.FindStringSubmatch(s)
	if m == nil {
		return 0
	}
	return parseSizeValue(m[1], m[2])
}

func parseSizeValue(numStr, unit string) int {
	val, _ := strconv.ParseFloat(numStr, 64)
	switch strings.ToUpper(unit) {
	case "K":
		return int(val * 1000)
	case "M":
		return int(val * 1000000)
	}
	return int(val)
}

func parseDollarAmount(s string) float64 {
	m := dollarRe.FindStringSubmatch(s)
	if m == nil {
		return 0
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	return v
}

// --- OpenAI ---

func fetchOpenAIModels(ctx context.Context) ([]ModelPricing, error) {
	body, err := providerHTTPGet(ctx, openAISourceURL)
	if err != nil {
		return nil, err
	}
	return parseOpenAIText(body)
}

var (
	openAIPricingRowRe = regexp.MustCompile(`(?m)^\|\s*([a-z0-9][\w.-]*)\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|`)
	openAISnapshotRe   = regexp.MustCompile(`(?m)^###\s+([\w.-]+)`)
	openAICtxWindowRe  = regexp.MustCompile(`(?m)^-\s*Context window size:\s*(\d+)`)
	openAIMaxOutputRe  = regexp.MustCompile(`(?m)^-\s*Maximum output tokens:\s*(\d+)`)
)

func parseOpenAIText(text string) ([]ModelPricing, error) {
	models := make(map[string]*ModelPricing)

	// Parse pricing table (## Text tokens section)
	if idx := strings.Index(text, "## Text tokens"); idx >= 0 {
		section := text[idx:]
		if end := strings.Index(section[1:], "\n## "); end >= 0 {
			section = section[:end+1]
		}
		for _, line := range strings.Split(section, "\n") {
			if strings.Contains(line, "(batch)") {
				continue
			}
			m := openAIPricingRowRe.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			name := strings.TrimSpace(m[1])
			input, _ := strconv.ParseFloat(m[2], 64)
			output, _ := strconv.ParseFloat(m[3], 64)
			models[name] = &ModelPricing{
				ID:               name,
				InputPricePer1M:  roundPrice(input),
				OutputPricePer1M: roundPrice(output),
			}
		}
	}

	// Parse model metadata sections for context window and max output
	parts := openAISnapshotRe.Split(text, -1)
	names := openAISnapshotRe.FindAllStringSubmatch(text, -1)
	for i, name := range names {
		section := parts[i+1]
		modelID := name[1]

		var ctxWindow, maxOutput int
		if m := openAICtxWindowRe.FindStringSubmatch(section); m != nil {
			ctxWindow, _ = strconv.Atoi(m[1])
		}
		if m := openAIMaxOutputRe.FindStringSubmatch(section); m != nil {
			maxOutput, _ = strconv.Atoi(m[1])
		}
		if ctxWindow == 0 && maxOutput == 0 {
			continue
		}
		if existing, ok := models[modelID]; ok {
			existing.ContextWindow = ctxWindow
			existing.MaxOutput = maxOutput
		} else {
			models[modelID] = &ModelPricing{
				ID:            modelID,
				ContextWindow: ctxWindow,
				MaxOutput:     maxOutput,
			}
		}
	}

	result := make([]ModelPricing, 0, len(models))
	for _, m := range models {
		result = append(result, *m)
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("no models found in OpenAI text")
	}
	return result, nil
}

// --- Anthropic ---

func fetchAnthropicModels(ctx context.Context) ([]ModelPricing, error) {
	body, err := providerHTTPGet(ctx, anthropicSourceURL)
	if err != nil {
		return nil, err
	}
	return parseAnthropicHTML(body)
}

var anthropicPriceRe = regexp.MustCompile(`(?s)\$\s*([\d.]+)\s*/\s*input.*?\$\s*([\d.]+)\s*/\s*output`)

func parseAnthropicHTML(html string) ([]ModelPricing, error) {
	tables := parseHTMLTables(html)

	var all []ModelPricing
	for _, table := range tables {
		all = append(all, parseAnthropicTable(table)...)
	}
	if len(all) == 0 {
		return nil, fmt.Errorf("no models found in Anthropic HTML")
	}
	return all, nil
}

// parseAnthropicTable handles the transposed table format where models are
// columns and attributes (API ID, Pricing, Context window, Max output) are rows.
func parseAnthropicTable(rows [][]string) []ModelPricing {
	if len(rows) < 2 {
		return nil
	}

	var modelIDs []string
	var aliases []string
	inputPrices := make(map[int]float64)
	outputPrices := make(map[int]float64)
	contextWindows := make(map[int]int)
	maxOutputs := make(map[int]int)

	for _, row := range rows {
		if len(row) < 2 {
			continue
		}
		label := strings.ToLower(row[0])

		switch {
		case strings.Contains(label, "api id") && !strings.Contains(label, "alias"):
			for _, cell := range row[1:] {
				modelIDs = append(modelIDs, strings.TrimSpace(cell))
			}

		case strings.Contains(label, "api") && strings.Contains(label, "alias"):
			for _, cell := range row[1:] {
				aliases = append(aliases, strings.TrimSpace(cell))
			}

		case strings.Contains(label, "pricing") || (strings.Contains(label, "price") && strings.Contains(label, "input")):
			for j, cell := range row[1:] {
				if m := anthropicPriceRe.FindStringSubmatch(cell); m != nil {
					inputPrices[j], _ = strconv.ParseFloat(m[1], 64)
					outputPrices[j], _ = strconv.ParseFloat(m[2], 64)
				}
			}

		case strings.Contains(label, "context window"):
			for j, cell := range row[1:] {
				if v := parseTokenCount(cell); v > 0 {
					contextWindows[j] = v
				}
			}

		case strings.Contains(label, "max output"):
			for j, cell := range row[1:] {
				if v := parseTokenCount(cell); v > 0 {
					maxOutputs[j] = v
				}
			}
		}
	}

	if len(modelIDs) == 0 {
		return nil
	}

	isDash := func(s string) bool {
		return s == "" || s == "—" || s == "-" || s == "–"
	}

	buildModel := func(id string, idx int) ModelPricing {
		return ModelPricing{
			ID:               id,
			InputPricePer1M:  inputPrices[idx],
			OutputPricePer1M: outputPrices[idx],
			ContextWindow:    contextWindows[idx],
			MaxOutput:        maxOutputs[idx],
		}
	}

	var models []ModelPricing
	for i, id := range modelIDs {
		if isDash(id) {
			continue
		}
		models = append(models, buildModel(id, i))

		// Also add alias if different from ID
		if i < len(aliases) && !isDash(aliases[i]) && aliases[i] != id {
			models = append(models, buildModel(aliases[i], i))
		}
	}
	return models
}

// --- Gemini ---

func fetchGeminiModels(ctx context.Context) ([]ModelPricing, error) {
	body, err := providerHTTPGet(ctx, geminiSourceURL)
	if err != nil {
		return nil, err
	}
	return parseGeminiHTML(body)
}

var geminiModelRe = regexp.MustCompile(`(?i)Gemini\s+([\d.]+)\s+([\w-]+)`)

func parseGeminiHTML(html string) ([]ModelPricing, error) {
	text := stripHTMLTags(html)

	indices := geminiModelRe.FindAllStringIndex(text, -1)
	names := geminiModelRe.FindAllStringSubmatch(text, -1)
	if len(indices) == 0 {
		return nil, fmt.Errorf("no models found in Gemini HTML")
	}

	seen := make(map[string]bool)
	var models []ModelPricing

	for i, name := range names {
		version := name[1]
		variant := strings.ToLower(name[2])

		// Skip non-LLM models
		if variant == "nano" || variant == "embeddings" || variant == "embedding" {
			continue
		}
		// Check if "Image" or "TTS" follows immediately (image/audio generation variant)
		afterMatch := ""
		if indices[i][1] < len(text) {
			end := indices[i][1] + 20
			if end > len(text) {
				end = len(text)
			}
			afterMatch = strings.TrimSpace(text[indices[i][1]:end])
		}
		if strings.HasPrefix(afterMatch, "Image") || strings.HasPrefix(afterMatch, "TTS") {
			continue
		}

		modelID := "gemini-" + version + "-" + variant
		if seen[modelID] {
			continue
		}

		// Section: text from end of this match to start of next
		start := indices[i][1]
		end := len(text)
		if i+1 < len(indices) {
			end = indices[i+1][0]
		}
		section := text[start:end]

		inputPrice := findFirstPrice(section, "input")
		outputPrice := findFirstPrice(section, "output")
		if inputPrice <= 0 && outputPrice <= 0 {
			continue
		}

		seen[modelID] = true
		models = append(models, ModelPricing{
			ID:               modelID,
			InputPricePer1M:  roundPrice(inputPrice),
			OutputPricePer1M: roundPrice(outputPrice),
		})
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no pricing found in Gemini HTML")
	}
	return models, nil
}

// findFirstPrice finds the first dollar amount near a keyword in a text section.
func findFirstPrice(section, keyword string) float64 {
	lower := strings.ToLower(section)
	idx := strings.Index(lower, keyword)
	if idx < 0 {
		return 0
	}
	end := idx + 100
	if end > len(section) {
		end = len(section)
	}
	m := dollarRe.FindStringSubmatch(section[idx:end])
	if m == nil {
		return 0
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	return v
}

// --- Grok ---

func fetchGrokModels(ctx context.Context) ([]ModelPricing, error) {
	body, err := providerHTTPGet(ctx, grokSourceURL)
	if err != nil {
		return nil, err
	}
	return parseGrokPage(body)
}

// Grok model data is embedded in the page as escaped JSON within the RSC payload.
// Each model entry has fields like:
//
//	"name":"grok-3","promptTextTokenPrice":"$n30000","completionTextTokenPrice":"$n150000","maxPromptLength":131072
//
// Prices use "$nXXXXX" format where the integer / 10000 = dollars per 1M tokens.
// Models appear once per cluster (us-east-1, eu-west-1, etc.) so we deduplicate.
var (
	grokModelRe       = regexp.MustCompile(`"name":"(grok-[^"]+)"`)
	grokInputPriceRe  = regexp.MustCompile(`"promptTextTokenPrice":"\$n(\d+)"`)
	grokOutputPriceRe = regexp.MustCompile(`"completionTextTokenPrice":"\$n(\d+)"`)
	grokContextRe     = regexp.MustCompile(`"maxPromptLength":(\d+)`)
)

func parseGrokPage(html string) ([]ModelPricing, error) {
	// Unescape the RSC payload: \" → "
	text := strings.ReplaceAll(html, `\"`, `"`)

	// Find all model entries with their surrounding context
	nameMatches := grokModelRe.FindAllStringSubmatchIndex(text, -1)
	if len(nameMatches) == 0 {
		return nil, fmt.Errorf("no models found in Grok page")
	}

	seen := make(map[string]bool)
	var models []ModelPricing

	for _, loc := range nameMatches {
		name := text[loc[2]:loc[3]]

		// Skip image/video models
		if strings.Contains(name, "imagine") || strings.Contains(name, "video") {
			continue
		}
		if seen[name] {
			continue
		}

		// Extract data from the region after the name match (fields follow in sequence)
		start := loc[1]
		end := start + 500
		if end > len(text) {
			end = len(text)
		}
		region := text[start:end]

		var inputPrice, outputPrice float64
		var contextWindow int

		if m := grokInputPriceRe.FindStringSubmatch(region); m != nil {
			v, _ := strconv.ParseFloat(m[1], 64)
			inputPrice = roundPrice(v / 10000)
		}
		if m := grokOutputPriceRe.FindStringSubmatch(region); m != nil {
			v, _ := strconv.ParseFloat(m[1], 64)
			outputPrice = roundPrice(v / 10000)
		}
		if m := grokContextRe.FindStringSubmatch(region); m != nil {
			contextWindow, _ = strconv.Atoi(m[1])
		}

		if inputPrice > 0 || outputPrice > 0 || contextWindow > 0 {
			seen[name] = true
			models = append(models, ModelPricing{
				ID:               name,
				InputPricePer1M:  inputPrice,
				OutputPricePer1M: outputPrice,
				ContextWindow:    contextWindow,
			})
		}
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no models found in Grok page")
	}
	return models, nil
}
