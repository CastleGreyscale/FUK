import React, { useState, useEffect } from 'react';
import { Camera, Film, Image, Settings, Clock, Zap, Loader2, CheckCircle, XCircle, Download, Eye } from 'lucide-react';

const FUKGenerationUI = () => {
  const [activeTab, setActiveTab] = useState('image');
  const [activeGenerations, setActiveGenerations] = useState([]);
  const [imageHistory, setImageHistory] = useState([]);
  const [videoHistory, setVideoHistory] = useState([]);
  const [config, setConfig] = useState(null);
  
  // API base URL
  const API_URL = 'http://localhost:8000/api';
  
  // Load config and history on mount
  useEffect(() => {
    loadConfig();
    loadHistory();
    
    // Poll for active generations
    const interval = setInterval(() => {
      refreshActiveGenerations();
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  const loadConfig = async () => {
    try {
      const [defaultsRes, modelsRes] = await Promise.all([
        fetch(`${API_URL}/config/defaults`),
        fetch(`${API_URL}/config/models`)
      ]);
      
      const defaults = await defaultsRes.json();
      const models = await modelsRes.json();
      
      setConfig({ defaults, models });
    } catch (error) {
      console.error('Failed to load config:', error);
    }
  };
  
  const loadHistory = async () => {
    try {
      const [imagesRes, videosRes] = await Promise.all([
        fetch(`${API_URL}/history/images?limit=50`),
        fetch(`${API_URL}/history/videos?limit=50`)
      ]);
      
      const images = await imagesRes.json();
      const videos = await videosRes.json();
      
      setImageHistory(images.generations || []);
      setVideoHistory(videos.generations || []);
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };
  
  const refreshActiveGenerations = async () => {
    try {
      const res = await fetch(`${API_URL}/status`);
      const data = await res.json();
      setActiveGenerations(data.active || []);
    } catch (error) {
      console.error('Failed to refresh active generations:', error);
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
      {/* Header */}
      <header className="border-b border-gray-700 bg-gray-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">FUK Generation Pipeline</h1>
                <p className="text-sm text-gray-400">Qwen Image × Wan Video</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {activeGenerations.length > 0 && (
                <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/20 border border-purple-500/30 rounded-full">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">{activeGenerations.length} active</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>
      
      {/* Tabs */}
      <div className="border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-2">
            <TabButton
              active={activeTab === 'image'}
              onClick={() => setActiveTab('image')}
              icon={<Camera className="w-4 h-4" />}
              label="Image Generation"
            />
            <TabButton
              active={activeTab === 'video'}
              onClick={() => setActiveTab('video')}
              icon={<Film className="w-4 h-4" />}
              label="Video Generation"
            />
            <TabButton
              active={activeTab === 'gallery'}
              onClick={() => setActiveTab('gallery')}
              icon={<Image className="w-4 h-4" />}
              label="Gallery"
            />
            <TabButton
              active={activeTab === 'queue'}
              onClick={() => setActiveTab('queue')}
              icon={<Clock className="w-4 h-4" />}
              label="Queue"
              badge={activeGenerations.length}
            />
          </div>
        </div>
      </div>
      
      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'image' && <ImageGenerationTab config={config} onGenerate={loadHistory} />}
        {activeTab === 'video' && <VideoGenerationTab config={config} onGenerate={loadHistory} />}
        {activeTab === 'gallery' && <GalleryTab imageHistory={imageHistory} videoHistory={videoHistory} />}
        {activeTab === 'queue' && <QueueTab generations={activeGenerations} />}
      </main>
    </div>
  );
};

// Tab Button Component
const TabButton = ({ active, onClick, icon, label, badge }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
      active
        ? 'border-purple-500 text-purple-400'
        : 'border-transparent text-gray-400 hover:text-gray-300'
    }`}
  >
    {icon}
    <span className="font-medium">{label}</span>
    {badge > 0 && (
      <span className="px-2 py-0.5 bg-purple-500 text-white text-xs rounded-full">
        {badge}
      </span>
    )}
  </button>
);

// Image Generation Tab
const ImageGenerationTab = ({ config, onGenerate }) => {
  const defaults = config?.defaults || {};
  
  const [formData, setFormData] = useState({
    prompt: '',
    model: 'qwen_image',
    negative_prompt: defaults.negative_prompt || '',
    width: defaults.image_size?.[0] || 1024,
    height: defaults.image_size?.[1] || 1024,
    steps: defaults.infer_steps || 20,
    guidance_scale: defaults.guidance_scale || 2.1,
    flow_shift: defaults.flow_shift || 2.1,
    seed: null,
    lora: null,
    lora_multiplier: defaults.lora_multiplier || 1.0,
    blocks_to_swap: defaults.blocks_to_swap || 10,
    output_format: 'png'
  });
  
  const [generating, setGenerating] = useState(false);
  const [generationId, setGenerationId] = useState(null);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  
  // Monitor progress
  useEffect(() => {
    if (!generationId) return;
    
    const eventSource = new EventSource(`http://localhost:8000/api/progress/${generationId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data);
      
      if (data.status === 'complete') {
        setResult(data);
        setGenerating(false);
        onGenerate();
        eventSource.close();
      } else if (data.status === 'failed') {
        setGenerating(false);
        eventSource.close();
      }
    };
    
    return () => eventSource.close();
  }, [generationId]);
  
  const handleGenerate = async () => {
    setGenerating(true);
    setProgress(null);
    setResult(null);
    
    try {
      const res = await fetch('http://localhost:8000/api/generate/image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await res.json();
      setGenerationId(data.generation_id);
    } catch (error) {
      console.error('Generation failed:', error);
      setGenerating(false);
    }
  };
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left: Controls */}
      <div className="space-y-6">
        <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Generation Parameters
          </h2>
          
          <div className="space-y-4">
            {/* Prompt */}
            <div>
              <label className="block text-sm font-medium mb-2">Prompt</label>
              <textarea
                value={formData.prompt}
                onChange={(e) => setFormData({ ...formData, prompt: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                rows={3}
                placeholder="A noir film style image. An apple on a wooden table."
              />
            </div>
            
            {/* Model */}
            <div>
              <label className="block text-sm font-medium mb-2">Model</label>
              <select
                value={formData.model}
                onChange={(e) => setFormData({ ...formData, model: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg"
              >
                <option value="qwen_image">Qwen Image</option>
                <option value="qwen_image_2509_edit">Qwen Edit 2509</option>
              </select>
            </div>
            
            {/* Dimensions */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Width</label>
                <input
                  type="number"
                  value={formData.width}
                  onChange={(e) => setFormData({ ...formData, width: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg"
                  step={64}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Height</label>
                <input
                  type="number"
                  value={formData.height}
                  onChange={(e) => setFormData({ ...formData, height: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg"
                  step={64}
                />
              </div>
            </div>
            
            {/* Steps */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Steps: {formData.steps}
              </label>
              <input
                type="range"
                value={formData.steps}
                onChange={(e) => setFormData({ ...formData, steps: parseInt(e.target.value) })}
                min={10}
                max={50}
                className="w-full"
              />
            </div>
            
            {/* Guidance Scale */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Guidance Scale: {formData.guidance_scale}
              </label>
              <input
                type="range"
                value={formData.guidance_scale}
                onChange={(e) => setFormData({ ...formData, guidance_scale: parseFloat(e.target.value) })}
                min={1}
                max={10}
                step={0.1}
                className="w-full"
              />
            </div>
            
            {/* Flow Shift */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Flow Shift: {formData.flow_shift}
              </label>
              <input
                type="range"
                value={formData.flow_shift}
                onChange={(e) => setFormData({ ...formData, flow_shift: parseFloat(e.target.value) })}
                min={1}
                max={5}
                step={0.1}
                className="w-full"
              />
              <p className="text-xs text-gray-400 mt-1">
                Lower (2.0-2.2) = more natural, Higher (3.0+) = more detail
              </p>
            </div>
            
            {/* Seed */}
            <div>
              <label className="block text-sm font-medium mb-2">Seed (optional)</label>
              <input
                type="number"
                value={formData.seed || ''}
                onChange={(e) => setFormData({ ...formData, seed: e.target.value ? parseInt(e.target.value) : null })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg"
                placeholder="Random"
              />
            </div>
            
            {/* LoRA */}
            {config?.models?.loras && config.models.loras.length > 0 && (
              <>
                <div>
                  <label className="block text-sm font-medium mb-2">LoRA</label>
                  <select
                    value={formData.lora || ''}
                    onChange={(e) => setFormData({ ...formData, lora: e.target.value || null })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg"
                  >
                    <option value="">None</option>
                    {config.models.loras.map(lora => (
                      <option key={lora} value={lora}>{lora}</option>
                    ))}
                  </select>
                </div>
                
                {formData.lora && (
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      LoRA Strength: {formData.lora_multiplier}
                    </label>
                    <input
                      type="range"
                      value={formData.lora_multiplier}
                      onChange={(e) => setFormData({ ...formData, lora_multiplier: parseFloat(e.target.value) })}
                      min={0}
                      max={2}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                )}
              </>
            )}
            
            {/* Output Format */}
            <div>
              <label className="block text-sm font-medium mb-2">Output Format</label>
              <select
                value={formData.output_format}
                onChange={(e) => setFormData({ ...formData, output_format: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg"
              >
                <option value="png">PNG Only</option>
                <option value="exr">EXR Only</option>
                <option value="both">Both PNG + EXR</option>
              </select>
            </div>
          </div>
          
          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={generating || !formData.prompt}
            className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed rounded-lg font-medium transition-all flex items-center justify-center gap-2"
          >
            {generating ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Generate Image
              </>
            )}
          </button>
        </div>
      </div>
      
      {/* Right: Preview & Progress */}
      <div className="space-y-6">
        {/* Progress */}
        {progress && (
          <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold">Generation Progress</h3>
              <span className="text-sm text-gray-400">{progress.phase}</span>
            </div>
            
            <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
              <div
                className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(progress.progress || 0) * 100}%` }}
              />
            </div>
            
            <div className="text-sm text-gray-400">
              {Math.round((progress.progress || 0) * 100)}% complete
            </div>
          </div>
        )}
        
        {/* Result */}
        {result && result.outputs && (
          <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                Generation Complete
              </h3>
            </div>
            
            {result.outputs.png && (
              <div className="space-y-4">
                <img
                  src={`http://localhost:8000${result.outputs.png.replace('outputs/', '/api/file/')}`}
                  alt="Generated"
                  className="w-full rounded-lg border border-gray-700"
                />
                
                <div className="flex gap-2">
                  <a
                    href={`http://localhost:8000${result.outputs.png.replace('outputs/', '/api/file/')}`}
                    download
                    className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download PNG
                  </a>
                  
                  {result.outputs.exr && (
                    <a
                      href={`http://localhost:8000${result.outputs.exr.replace('outputs/', '/api/file/')}`}
                      download
                      className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2"
                    >
                      <Download className="w-4 h-4" />
                      Download EXR
                    </a>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Placeholder */}
        {!progress && !result && (
          <div className="bg-gray-800/50 rounded-lg border border-gray-700 border-dashed p-12 flex items-center justify-center">
            <div className="text-center text-gray-500">
              <Camera className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Generated image will appear here</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Video Generation Tab (simplified version - similar structure)
const VideoGenerationTab = ({ config, onGenerate }) => {
  const defaults = config?.defaults || {};
  
  const [formData, setFormData] = useState({
    prompt: '',
    task: 'i2v-14B',
    width: 832,
    height: 480,
    video_length: 81,
    steps: 20,
    guidance_scale: 5.0,
    flow_shift: 5.0,
    seed: null,
    export_exr: false
  });
  
  const [generating, setGenerating] = useState(false);
  
  const handleGenerate = async () => {
    setGenerating(true);
    
    try {
      const res = await fetch('http://localhost:8000/api/generate/video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await res.json();
      // Handle video generation similar to image
    } catch (error) {
      console.error('Video generation failed:', error);
    } finally {
      setGenerating(false);
    }
  };
  
  return (
    <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-6">
      <h2 className="text-lg font-semibold mb-4">Video Generation</h2>
      <p className="text-gray-400">Video generation UI - similar to image tab with video-specific controls</p>
      
      <button
        onClick={handleGenerate}
        disabled={generating}
        className="mt-4 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg font-medium"
      >
        {generating ? 'Generating...' : 'Generate Video'}
      </button>
    </div>
  );
};

// Gallery Tab
const GalleryTab = ({ imageHistory, videoHistory }) => {
  const [view, setView] = useState('images');
  
  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <button
          onClick={() => setView('images')}
          className={`px-4 py-2 rounded-lg ${view === 'images' ? 'bg-purple-500' : 'bg-gray-700'}`}
        >
          Images ({imageHistory.length})
        </button>
        <button
          onClick={() => setView('videos')}
          className={`px-4 py-2 rounded-lg ${view === 'videos' ? 'bg-purple-500' : 'bg-gray-700'}`}
        >
          Videos ({videoHistory.length})
        </button>
      </div>
      
      {view === 'images' && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {imageHistory.map((gen, idx) => (
            <div key={idx} className="bg-gray-800/50 rounded-lg border border-gray-700 overflow-hidden">
              {gen.thumbnail && (
                <img
                  src={`http://localhost:8000${gen.thumbnail}`}
                  alt={gen.prompt}
                  className="w-full aspect-square object-cover"
                />
              )}
              <div className="p-3">
                <p className="text-sm text-gray-400 line-clamp-2">{gen.prompt}</p>
                <p className="text-xs text-gray-500 mt-1">Seed: {gen.seed}</p>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {view === 'videos' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {videoHistory.map((gen, idx) => (
            <div key={idx} className="bg-gray-800/50 rounded-lg border border-gray-700 p-4">
              <p className="text-sm text-gray-400 mb-2">{gen.prompt}</p>
              {gen.video && (
                <video
                  src={`http://localhost:8000${gen.video}`}
                  controls
                  className="w-full rounded"
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Queue Tab
const QueueTab = ({ generations }) => {
  return (
    <div className="space-y-4">
      {generations.length === 0 ? (
        <div className="bg-gray-800/50 rounded-lg border border-gray-700 border-dashed p-12 text-center text-gray-500">
          <Clock className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No active generations</p>
        </div>
      ) : (
        generations.map((gen) => (
          <div key={gen.id} className="bg-gray-800/50 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium capitalize">{gen.type} Generation</span>
              <span className="text-sm text-gray-400">{gen.phase}</span>
            </div>
            
            <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
              <div
                className="bg-purple-500 h-2 rounded-full transition-all"
                style={{ width: `${gen.progress * 100}%` }}
              />
            </div>
            
            <span className="text-xs text-gray-500">ID: {gen.id}</span>
          </div>
        ))
      )}
    </div>
  );
};

export default FUKGenerationUI;
