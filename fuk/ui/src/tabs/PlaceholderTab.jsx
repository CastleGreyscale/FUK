/**
 * Placeholder Tab Component
 * Used for tabs that aren't implemented yet
 * Refactored: All inline styles moved to CSS classes
 */

import Footer from '../components/Footer';

export default function PlaceholderTab({ 
  title, 
  description, 
  icon: Icon, 
  activeTab, 
  setActiveTab 
}) {
  return (
    <>
      <div className="fuk-preview-area">
        <div className="fuk-placeholder-card fuk-placeholder-card--40">
          <div className="fuk-placeholder">
            {Icon && <Icon className="fuk-placeholder-icon fuk-placeholder-icon--faded" />}
            <p className="fuk-placeholder-text">{title}</p>
            {description && (
              <p className="fuk-placeholder-subtext">{description}</p>
            )}
          </div>
        </div>
      </div>

      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={false}
        progress={null}
        elapsedSeconds={0}
        onGenerate={() => {}}
        onCancel={() => {}}
        canGenerate={false}
        generateLabel="Not Available"
      />
    </>
  );
}