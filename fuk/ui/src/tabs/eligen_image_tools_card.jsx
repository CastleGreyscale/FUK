/*
 * ImageTab.jsx — Image Tools Card replacement
 * 
 * REPLACES the entire {/* Image Tools Card *\/} section (lines ~360-419)
 * 
 * What changed:
 *   - Added EliGen condition alongside edit_image/context_image
 *   - When model supports "eligen": shows path input + browse buttons
 *   - Browse Folder uses /api/browser/directory (existing endpoint)
 *   - Browse File uses /api/browser/open with filter for .psd/.ora (existing endpoint)
 *   - Empty state updated to mention EliGen models
 * 
 * ALSO ADD to initialDefaults (inside useMemo):
 *   eligen_source: imageDefaults.eligen_source ?? '',
 * 
 * ALSO ADD to payload in handleGenerate:
 *   eligen_source: formData.eligen_source || null,
 * 
 * ALSO ADD this import at the top with the other icon imports:
 *   FolderOpen  (if not already imported)
 */

          {/* Image Tools Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Image Tools</h3>
            
            {/* Control Images — for edit and control-union models */}
            {(modelSupports(formData.model, 'edit_image') || modelSupports(formData.model, 'context_image')) ? (
              <>
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">
                    Control Images ({formData.control_image_paths.length})
                  </label>
                  <MediaUploader
                    images={formData.control_image_paths}
                    onImagesChange={handleImagesChange}
                    disabled={generating}
                  />
                </div>
                
                {formData.control_image_paths.length > 0 && (
                  <div className="fuk-form-group-compact fuk-mt-4">
                    <label className="fuk-label">Edit Strength</label>
                    <div className="fuk-input-inline">
                      <input
                        type="range"
                        className="fuk-input fuk-input--flex-2"
                        value={formData.edit_strength || 0.85}
                        onChange={(e) => setFormData({...formData, edit_strength: parseFloat(e.target.value)})}
                        min={0}
                        max={1}
                        step={0.05}
                      />
                      <input
                        type="number"
                        className="fuk-input fuk-input--w-80"
                        value={formData.edit_strength || 0.85}
                        onChange={(e) => setFormData({...formData, edit_strength: parseFloat(e.target.value)})}
                        step={0.05}
                        min={0}
                        max={1}
                      />
                    </div>
                    <p className="fuk-help-text fuk-mt-1">
                      Controls how much the control images influence generation. 
                      1.0 = full strength, 0.0 = minimal influence.
                    </p>
                  </div>
                )}
                
                <p className="fuk-help-text">
                  Upload one or more images to guide the generation.
                </p>
              </>

            /* EliGen — entity composition masks */
            ) : modelSupports(formData.model, 'eligen') ? (
              <>
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">Entity Masks</label>
                  <div className="fuk-input-inline">
                    <input
                      type="text"
                      className="fuk-input fuk-input--flex-2"
                      value={formData.eligen_source || ''}
                      onChange={(e) => setFormData({...formData, eligen_source: e.target.value})}
                      placeholder="/path/to/masks/ or .psd / .ora file"
                      disabled={generating}
                    />
                  </div>
                  <div className="fuk-input-inline fuk-mt-2" style={{ gap: '0.5rem' }}>
                    <button
                      className="fuk-btn fuk-btn-secondary"
                      style={{ flex: 1 }}
                      disabled={generating}
                      onClick={async () => {
                        try {
                          const res = await fetch('/api/browser/directory', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ title: 'Select EliGen Masks Folder' }),
                          });
                          const data = await res.json();
                          if (data.success && data.directory) {
                            setFormData(prev => ({ ...prev, eligen_source: data.directory }));
                          }
                        } catch (err) { console.error('Browse folder failed:', err); }
                      }}
                    >
                      Browse Folder
                    </button>
                    <button
                      className="fuk-btn fuk-btn-secondary"
                      style={{ flex: 1 }}
                      disabled={generating}
                      onClick={async () => {
                        try {
                          const res = await fetch('/api/browser/open', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              title: 'Select EliGen Mask File (.psd / .ora)',
                              multiple: false,
                              detect_sequences: false,
                              filter: 'all',
                            }),
                          });
                          const data = await res.json();
                          if (data.success && data.files?.length > 0) {
                            setFormData(prev => ({ ...prev, eligen_source: data.files[0].path }));
                          }
                        } catch (err) { console.error('Browse file failed:', err); }
                      }}
                    >
                      Browse File
                    </button>
                  </div>
                </div>
                
                {formData.eligen_source && (
                  <div className="fuk-form-group-compact fuk-mt-2">
                    <div style={{ 
                      padding: '0.5rem 0.75rem',
                      background: 'rgba(255,255,255,0.03)',
                      borderRadius: '4px',
                      fontSize: '0.8rem',
                      color: 'var(--fuk-text-secondary)',
                      wordBreak: 'break-all',
                    }}>
                      {formData.eligen_source}
                    </div>
                  </div>
                )}
                
                <p className="fuk-help-text fuk-mt-2">
                  Point to a folder of mask PNGs (filename = entity prompt) or a 
                  layered file (.psd / .ora — layer name = entity prompt).
                  Paint white where each entity goes, black elsewhere.
                </p>
              </>

            ) : (
              <div className="fuk-empty-state">
                <Camera className="fuk-empty-state-icon" />
                <p className="fuk-empty-state-text">
                  Select an Edit, Control, or EliGen model<br />to enable image tools
                </p>
              </div>
            )}
          </div>
