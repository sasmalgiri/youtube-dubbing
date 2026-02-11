import './globals.css';
import type { Metadata } from 'next';
import Sidebar from '@/components/Sidebar';

export const metadata: Metadata = {
    title: 'VoiceDub - YouTube Video Dubbing',
    description: 'Dub YouTube videos into Hindi with AI-powered voice synthesis.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en">
            <body className="bg-background text-text-primary">
                <div className="flex min-h-screen">
                    <Sidebar />
                    <main className="flex-1 ml-64">
                        {children}
                    </main>
                </div>
            </body>
        </html>
    );
}
