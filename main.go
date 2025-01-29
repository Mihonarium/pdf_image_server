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

type Config struct {
    PDFs            map[string]string `yaml:"pdfs"`
    Port            int              `yaml:"port"`
    CacheDir        string           `yaml:"cache_dir"`
    ShutdownTimeout time.Duration    `yaml:"shutdown_timeout"`
}

type Server struct {
    config     *Config
    pdfs       map[string]*PDF
    httpServer *http.Server
}

type PageInfo struct {
    img        image.Image
    lastAccess time.Time
}

type PDF struct {
    path       string
    pages      map[int]*PageInfo
    totalPages int
    mu         sync.RWMutex
    hash       string
}

const (
    // Pages unused for this duration will be unloaded
    pageUnloadThreshold = 10 * time.Minute
    // Check for unused pages every this duration
    cleanupInterval = 5 * time.Minute
    // Maximum number of pages to keep in memory per PDF
    maxPagesInMemory = 100
)

func NewServer(config *Config) (*Server, error) {
    if err := validateConfig(config); err != nil {
        return nil, fmt.Errorf("invalid configuration: %w", err)
    }

    s := &Server{
        config: config,
        pdfs:   make(map[string]*PDF),
    }

    if err := s.initializeCache(); err != nil {
        return nil, fmt.Errorf("failed to initialize cache: %w", err)
    }

    if err := s.loadPDFs(); err != nil {
        return nil, fmt.Errorf("failed to load PDFs: %w", err)
    }

    // Start page cleanup routine
    go s.startPageCleanup()

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

func (s *Server) initializeCache() error {
    return os.MkdirAll(s.config.CacheDir, 0755)
}

func (s *Server) loadPDFs() error {
    errs := make(chan error, len(s.config.PDFs))
    var wg sync.WaitGroup

    for id, path := range s.config.PDFs {
        wg.Add(1)
        go func(id, path string) {
            defer wg.Done()
            if err := s.loadPDF(id, path); err != nil {
                errs <- fmt.Errorf("failed to load PDF %s: %w", id, err)
            }
        }(id, path)
    }

    wg.Wait()
    close(errs)

    var loadErrors []string
    for err := range errs {
        loadErrors = append(loadErrors, err.Error())
    }

    if len(loadErrors) > 0 {
        return fmt.Errorf("PDF loading errors: %s", strings.Join(loadErrors, "; "))
    }

    if len(s.pdfs) == 0 {
        return errors.New("no PDFs were successfully loaded")
    }

    return nil
}

func (s *Server) loadPDF(id, path string) error {
    hash, err := calculateFileHash(path)
    if err != nil {
        return fmt.Errorf("hash calculation failed: %w", err)
    }

    doc, err := fitz.New(path)
    if err != nil {
        return fmt.Errorf("document creation failed: %w", err)
    }
    defer doc.Close()

    pdf := &PDF{
        path:       path,
        pages:      make(map[int]*PageInfo),
        totalPages: doc.NumPage(),
        hash:       hash,
    }

    if err := s.loadPages(pdf, doc); err != nil {
        return fmt.Errorf("page loading failed: %w", err)
    }

    s.pdfs[id] = pdf
    return nil
}

func (s *Server) loadPages(pdf *PDF, doc *fitz.Document) error {
    numWorkers := runtime.NumCPU()
    jobs := make(chan int, pdf.totalPages)
    results := make(chan error, pdf.totalPages)
    var wg sync.WaitGroup

    // Start workers
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go s.pageWorker(doc, pdf, jobs, results, &wg)
    }

    // Send jobs
    go func() {
        for i := 0; i < pdf.totalPages; i++ {
            jobs <- i
        }
        close(jobs)
    }()

    // Wait for completion
    go func() {
        wg.Wait()
        close(results)
    }()

    // Collect errors
    var errs []string
    for err := range results {
        if err != nil {
            errs = append(errs, err.Error())
        }
    }

    if len(errs) > 0 {
        return fmt.Errorf("page errors: %s", strings.Join(errs, "; "))
    }

    return nil
}

func (s *Server) pageWorker(doc *fitz.Document, pdf *PDF, jobs <-chan int, results chan<- error, wg *sync.WaitGroup) {
    defer wg.Done()
    for pageNum := range jobs {
        results <- s.processPage(doc, pdf, pageNum)
    }
}

func (s *Server) processPage(doc *fitz.Document, pdf *PDF, pageNum int) error {
    cachePath := s.getCachePath(pdf.hash, pageNum)

    // Try loading from cache
    if img, err := s.loadFromCache(cachePath); err == nil {
        pdf.mu.Lock()
        pdf.pages[pageNum] = &PageInfo{
            img:        img,
            lastAccess: time.Now(),
        }
        pdf.mu.Unlock()
        return nil
    }

    // Render from PDF
    img, err := doc.Image(pageNum)
    if err != nil {
        return fmt.Errorf("page %d render failed: %w", pageNum, err)
    }

    // Save to cache
    if err := s.saveToCache(img, cachePath); err != nil {
        log.Printf("Cache save failed for page %d: %v", pageNum, err)
    }

    pdf.mu.Lock()
    pdf.pages[pageNum] = &PageInfo{
        img:        img,
        lastAccess: time.Now(),
    }
    pdf.mu.Unlock()

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

type CropParams struct {
    width    int
    height   int
    topLeftX int
    topLeftY int
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

    scale := 300.0 / 250.0
    return &CropParams{
        width:    int(float64(getIntParam("width")) * scale),
        height:   int(float64(getIntParam("height")) * scale),
        topLeftX: int(float64(getIntParam("top_left_x")) * scale),
        topLeftY: int(float64(getIntParam("top_left_y")) * scale),
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

    pdf.mu.RLock()
    pageInfo, ok := pdf.pages[page-1]
    if ok {
        pageInfo.lastAccess = time.Now()
    }
    pdf.mu.RUnlock()

    var img image.Image
    if !ok {
        // Try to load from cache if not in memory
        cachePath := s.getCachePath(pdf.hash, page-1)
        var err error
        img, err = s.loadFromCache(cachePath)
        if err != nil {
            http.Error(w, ErrPageUnavailable.Error(), http.StatusNotFound)
            return
        }
        
        pdf.mu.Lock()
        pdf.pages[page-1] = &PageInfo{
            img:        img,
            lastAccess: time.Now(),
        }
        pdf.mu.Unlock()
    } else {
        img = pageInfo.img
    }

    crop := parseCropParams(r.URL.Query())
    croppedImg, err := cropImage(img, crop)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "image/jpeg")
    w.Header().Set("Cache-Control", "public, max-age=86400")
    jpeg.Encode(w, croppedImg, &jpeg.Options{Quality: 90})
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

func (s *Server) startPageCleanup() {
    ticker := time.NewTicker(cleanupInterval)
    defer ticker.Stop()

    for range ticker.C {
        s.cleanupUnusedPages()
    }
}

func (s *Server) cleanupUnusedPages() {
    now := time.Now()
    threshold := now.Add(-pageUnloadThreshold)

    for _, pdf := range s.pdfs {
        pdf.mu.Lock()

        // Count pages and find oldest accesses
        type pageAccess struct {
            pageNum    int
            lastAccess time.Time
        }
        var accesses []pageAccess
        for pageNum, page := range pdf.pages {
            accesses = append(accesses, pageAccess{
                pageNum:    pageNum,
                lastAccess: page.lastAccess,
            })
        }

        // Sort by last access time, oldest first
        sort.Slice(accesses, func(i, j int) bool {
            return accesses[i].lastAccess.Before(accesses[j].lastAccess)
        })

        // Remove old pages and ensure we don't exceed maxPagesInMemory
        pagesRemoved := 0
        for _, pa := range accesses {
            if len(pdf.pages) <= maxPagesInMemory {
                // If we're under the limit, only remove very old pages
                if pa.lastAccess.After(threshold) {
                    break
                }
            }
            delete(pdf.pages, pa.pageNum)
            pagesRemoved++
        }

        pdf.mu.Unlock()

        if pagesRemoved > 0 {
            log.Printf("Unloaded %d pages from PDF %s", pagesRemoved, pdf.path)
        }
    }
}

func (s *Server) Shutdown(ctx context.Context) error {
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

    go func() {
        if err := server.Start(); err != nil {
            log.Printf("Server error: %v", err)
        }
    }()

    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
    <-sigChan

    shutdownCtx, cancel := context.WithTimeout(context.Background(), config.ShutdownTimeout)
    defer cancel()

    log.Printf("Shutting down server (timeout: %v)", config.ShutdownTimeout)
    if err := server.Shutdown(shutdownCtx); err != nil {
        log.Printf("Shutdown error: %v", err)
    }
}