package main

import (
    "context"
    "crypto/sha256"
    "encoding/hex"
    "errors"
    "fmt"
    "github.com/gen2brain/go-fitz"
    "github.com/gorilla/mux"
    "gopkg.in/yaml.v2"
    "image"
    "image/jpeg"
    "image/png"
    "io"
    "log"
    "net/http"
    "os"
    "os/signal"
    "path/filepath"
    "runtime"
    "sort"
    "strconv"
    "strings"
    "sync"
    "syscall"
    "time"
)

var (
    ErrInvalidPage     = errors.New("invalid page number")
    ErrPDFNotFound     = errors.New("PDF not found")
    ErrInvalidCrop     = errors.New("invalid crop parameters")
    ErrPageUnavailable = errors.New("page is not available")
)

const (
    pageUnloadThreshold = 10 * time.Minute
    cleanupInterval    = 5 * time.Minute
    maxPagesInMemory   = 100
    dpiScale           = 300.0 / 250.0
)

type Config struct {
    PDFs            map[string]string `yaml:"pdfs"`
    Port            int              `yaml:"port"`
    CacheDir        string           `yaml:"cache_dir"`
    ShutdownTimeout time.Duration    `yaml:"shutdown_timeout"`
}

type CropParams struct {
    width    int
    height   int
    topLeftX int
    topLeftY int
}

type PageInfo struct {
    img        image.Image
    lastAccess time.Time
}

type PDF struct {
    path       string
    pages      sync.Map // map[int]*PageInfo with concurrent access
    totalPages int
    hash       string
}

type Server struct {
    config     *Config
    pdfs       map[string]*PDF
    httpServer *http.Server
    done       chan struct{}
}

func NewServer(config *Config) (*Server, error) {
    if err := validateConfig(config); err != nil {
        return nil, fmt.Errorf("invalid configuration: %w", err)
    }

    s := &Server{
        config: config,
        pdfs:   make(map[string]*PDF),
        done:   make(chan struct{}),
    }

    if err := s.initialize(); err != nil {
        return nil, err
    }

    return s, nil
}

func validateConfig(config *Config) error {
    if config.CacheDir == "" {
        config.CacheDir = "cache"
    }
    if config.Port == 0 {
        config.Port = 8380
    }
    if config.ShutdownTimeout == 0 {
        config.ShutdownTimeout = 30 * time.Second
    }
    if len(config.PDFs) == 0 {
        return errors.New("no PDFs configured")
    }
    return nil
}

func (s *Server) initialize() error {
    if err := os.MkdirAll(s.config.CacheDir, 0755); err != nil {
        return fmt.Errorf("failed to create cache directory: %w", err)
    }

    if err := s.loadPDFs(); err != nil {
        return fmt.Errorf("failed to load PDFs: %w", err)
    }

    go s.startCleanupRoutine()
    return nil
}

func (s *Server) loadPDFs() error {
    type pdfLoadResult struct {
        id  string
        pdf *PDF
        err error
    }

    results := make(chan pdfLoadResult, len(s.config.PDFs))
    var wg sync.WaitGroup

    for id, path := range s.config.PDFs {
        wg.Add(1)
        go func(id, path string) {
            defer wg.Done()
            pdf, err := s.loadPDF(path)
            results <- pdfLoadResult{id, pdf, err}
        }(id, path)
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    var loadErrors []string
    for result := range results {
        if result.err != nil {
            loadErrors = append(loadErrors, fmt.Sprintf("PDF %s: %v", result.id, result.err))
            continue
        }
        s.pdfs[result.id] = result.pdf
    }

    if len(loadErrors) > 0 {
        return fmt.Errorf("PDF loading errors: %s", strings.Join(loadErrors, "; "))
    }

    if len(s.pdfs) == 0 {
        return errors.New("no PDFs were successfully loaded")
    }

    return nil
}

func (s *Server) loadPDF(path string) (*PDF, error) {
    hash, err := calculateFileHash(path)
    if err != nil {
        return nil, fmt.Errorf("hash calculation failed: %w", err)
    }

    doc, err := fitz.New(path)
    if err != nil {
        return nil, fmt.Errorf("document creation failed: %w", err)
    }
    defer doc.Close()

    pdf := &PDF{
        path:       path,
        totalPages: doc.NumPage(),
        hash:       hash,
    }

    if err := s.loadPages(pdf, doc); err != nil {
        return nil, fmt.Errorf("page loading failed: %w", err)
    }

    return pdf, nil
}

func (s *Server) loadPages(pdf *PDF, doc *fitz.Document) error {
    numWorkers := runtime.NumCPU()
    jobs := make(chan int, pdf.totalPages)
    errors := make(chan error, pdf.totalPages)

    var wg sync.WaitGroup
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go s.pageWorker(pdf, doc, jobs, errors, &wg)
    }

    go func() {
        for i := 0; i < pdf.totalPages; i++ {
            jobs <- i
        }
        close(jobs)
    }()

    go func() {
        wg.Wait()
        close(errors)
    }()

    var errs []string
    for err := range errors {
        if err != nil {
            errs = append(errs, err.Error())
        }
    }

    if len(errs) > 0 {
        return fmt.Errorf("page errors: %s", strings.Join(errs, "; "))
    }

    return nil
}

func (s *Server) pageWorker(pdf *PDF, doc *fitz.Document, jobs <-chan int, errors chan<- error, wg *sync.WaitGroup) {
    defer wg.Done()
    for pageNum := range jobs {
        errors <- s.processPage(pdf, doc, pageNum)
    }
}

func (s *Server) processPage(pdf *PDF, doc *fitz.Document, pageNum int) error {
    cachePath := s.getCachePath(pdf.hash, pageNum)

    // Try loading from cache first
    if img, err := s.loadFromCache(cachePath); err == nil {
        pdf.pages.Store(pageNum, &PageInfo{
            img:        img,
            lastAccess: time.Now(),
        })
        return nil
    }

    // Render from PDF if not in cache
    img, err := doc.Image(pageNum)
    if err != nil {
        return fmt.Errorf("page %d render failed: %w", pageNum, err)
    }

    // Save to cache
    if err := s.saveToCache(img, cachePath); err != nil {
        log.Printf("Warning: cache save failed for page %d: %v", pageNum, err)
    }

    pdf.pages.Store(pageNum, &PageInfo{
        img:        img,
        lastAccess: time.Now(),
    })

    return nil
}

func (s *Server) getCachePath(pdfHash string, pageNum int) string {
    return filepath.Join(s.config.CacheDir, fmt.Sprintf("%s-page%d.png", pdfHash, pageNum))
}

func (s *Server) loadFromCache(path string) (image.Image, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    return png.Decode(file)
}

func (s *Server) saveToCache(img image.Image, path string) error {
    tmpPath := path + ".tmp"
    file, err := os.Create(tmpPath)
    if err != nil {
        return err
    }

    if err := png.Encode(file, img); err != nil {
        file.Close()
        os.Remove(tmpPath)
        return err
    }

    if err := file.Close(); err != nil {
        os.Remove(tmpPath)
        return err
    }

    return os.Rename(tmpPath, path)
}

func (s *Server) startCleanupRoutine() {
    ticker := time.NewTicker(cleanupInterval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            s.cleanupUnusedPages()
        case <-s.done:
            return
        }
    }
}

func (s *Server) cleanupUnusedPages() {
    threshold := time.Now().Add(-pageUnloadThreshold)

    for _, pdf := range s.pdfs {
        var pages []struct {
            num   int
            info  *PageInfo
        }

        // Collect pages
        pdf.pages.Range(func(key, value interface{}) bool {
            pageNum := key.(int)
            pageInfo := value.(*PageInfo)
            pages = append(pages, struct {
                num   int
                info  *PageInfo
            }{pageNum, pageInfo})
            return true
        })

        // Sort by last access time
        sort.Slice(pages, func(i, j int) bool {
            return pages[i].info.lastAccess.Before(pages[j].info.lastAccess)
        })

        // Remove old pages and ensure we don't exceed maxPagesInMemory
        removed := 0
        for _, p := range pages {
            if len(pages)-removed <= maxPagesInMemory && p.info.lastAccess.After(threshold) {
                break
            }
            pdf.pages.Delete(p.num)
            removed++
        }

        if removed > 0 {
            log.Printf("Unloaded %d pages from PDF %s", removed, pdf.path)
        }
    }
}

func (s *Server) handleImage(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    pdfID := vars["id"]
    pageStr := strings.TrimSuffix(vars["page"], ".jpg")

    pdf, ok := s.pdfs[pdfID]
    if !ok {
        http.Error(w, ErrPDFNotFound.Error(), http.StatusNotFound)
        return
    }

    page, err := strconv.Atoi(pageStr)
    if err != nil || page < 1 || page > pdf.totalPages {
        http.Error(w, ErrInvalidPage.Error(), http.StatusBadRequest)
        return
    }

    pageNum := page - 1
    value, ok := pdf.pages.Load(pageNum)
    var img image.Image

    if !ok {
        // Try to load from cache
        cachePath := s.getCachePath(pdf.hash, pageNum)
        var err error
        if img, err = s.loadFromCache(cachePath); err != nil {
            http.Error(w, ErrPageUnavailable.Error(), http.StatusNotFound)
            return
        }
        pdf.pages.Store(pageNum, &PageInfo{
            img:        img,
            lastAccess: time.Now(),
        })
    } else {
        pageInfo := value.(*PageInfo)
        pageInfo.lastAccess = time.Now()
        img = pageInfo.img
    }

    crop := parseCropParams(r.URL.Query())
    result, err := cropImage(img, crop)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "image/jpeg")
    w.Header().Set("Cache-Control", "public, max-age=86400")
    jpeg.Encode(w, result, &jpeg.Options{Quality: 90})
}

func parseCropParams(query map[string][]string) *CropParams {
    getIntParam := func(name string) int {
        if values := query[name]; len(values) > 0 {
            if val, err := strconv.Atoi(values[0]); err == nil {
                return val
            }
        }
        return 0
    }

    return &CropParams{
        width:    int(float64(getIntParam("width")) * dpiScale),
        height:   int(float64(getIntParam("height")) * dpiScale),
        topLeftX: int(float64(getIntParam("top_left_x")) * dpiScale),
        topLeftY: int(float64(getIntParam("top_left_y")) * dpiScale),
    }
}

func cropImage(img image.Image, crop *CropParams) (image.Image, error) {
    bounds := img.Bounds()
    
    if crop.width == 0 && crop.height == 0 && crop.topLeftX == 0 && crop.topLeftY == 0 {
        return img, nil
    }

    if crop.topLeftX+crop.width > bounds.Max.X || crop.topLeftY+crop.height > bounds.Max.Y {
        return nil, ErrInvalidCrop
    }

    cropped := image.NewRGBA(image.Rect(0, 0, crop.width, crop.height))
    for y := 0; y < crop.height; y++ {
        for x := 0; x < crop.width; x++ {
            cropped.Set(x, y, img.At(crop.topLeftX+x, crop.topLeftY+y))
        }
    }

    return cropped, nil
}

func (s *Server) Start() error {
    router := mux.NewRouter()
    router.HandleFunc("/cropped/{id}-{page}", s.handleImage).Methods(http.MethodGet)

    s.httpServer = &http.Server{
        Addr:         fmt.Sprintf(":%d", s.config.Port),
        Handler:      router,
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
        IdleTimeout:  60 * time.Second,
    }

    log.Printf("Server starting on port %d", s.config.Port)
    if err := s.httpServer.ListenAndServe(); err != http.ErrServerClosed {
        return fmt.Errorf("server error: %w", err)
    }
    return nil
}

func (s *Server) Shutdown(ctx context.Context) error {
    close(s.done)
    return s.httpServer.Shutdown(ctx)
}

func calculateFileHash(path string) (string, error) {
    file, err := os.Open(path)
    if err != nil {
        return "", err
    }
    defer file.Close()

    hash := sha256.New()
    if _, err := io.Copy(hash, file); err != nil {
        return "", err
    }

    return hex.EncodeToString(hash.Sum(nil)), nil
}

func main() {
    if len(os.Args) != 2 {
        log.Fatal("Usage: program <config_file>")
    }

    log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds | log.Lshortfile)

    configData, err := os.ReadFile(os.Args[1])
    if err != nil {
        log.Fatalf("Failed to read config: %v", err)
    }

    var config Config
    if err := yaml.Unmarshal(configData, &config); err != nil {
        log.Fatalf("Failed to parse config: %v", err)
    }

    server, err := NewServer(&config)
    if err != nil {
        log.Fatalf("Failed to create server: %v", err)
    }

    // Handle graceful shutdown
    serverErr := make(chan error, 1)
    go func() {
        if err := server.Start(); err != nil {
            serverErr <- err
        }
    }()

    // Wait for interrupt or server error
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

    select {
    case err := <-serverErr:
        log.Printf("Server error: %v", err)
    case sig := <-sigChan:
        log.Printf("Received signal: %v", sig)
    }

    // Graceful shutdown
    shutdownCtx, cancel := context.WithTimeout(context.Background(), config.ShutdownTimeout)
    defer cancel()

    log.Printf("Shutting down server (timeout: %v)", config.ShutdownTimeout)
    if err := server.Shutdown(shutdownCtx); err != nil {
        log.Printf("Shutdown error: %v", err)
    }
}
