#!/usr/bin/env python
"""
Single job execution script for RFP Completeness Check.
This script runs a single RFP completeness check without starting the manager/worker threads.
"""

import sys
import argparse
from main import RFPCompleteness

def main():
    parser = argparse.ArgumentParser(description='Run a single RFP completeness check')
    parser.add_argument('--id', default='1', help='Job ID')
    parser.add_argument('--model', default='openai', choices=['openai', 'opensource'], help='Model to use')
    parser.add_argument('--rfp-url', required=True, help='URL to the RFP document')
    parser.add_argument('--ea-standard-url', required=True, help='URL to the EA Standard document')
    parser.add_argument('--output-tokens', default='1', help='Output tokens')
    parser.add_argument('--industry-standards', default='', help='Industry standards (comma-separated)')
    parser.add_argument('--ministry-compliances', default='', help='Ministry compliances (comma-separated)')
    parser.add_argument('--output-language', default='english', help='Output language')
    
    args = parser.parse_args()
    
    print(f"Running RFP Completeness Check with the following parameters:")
    print(f"  ID: {args.id}")
    print(f"  Model: {args.model}")
    print(f"  RFP URL: {args.rfp_url}")
    print(f"  EA Standard URL: {args.ea_standard_url}")
    print(f"  Output Language: {args.output_language}")
    print("-" * 80)
    
    try:
        # Create RFPCompleteness instance
        rfp_completeness = RFPCompleteness()
        
        # Run the completeness check
        id, result, error_msg_for_user, technical_error_msg = rfp_completeness.is_complete(
            id=args.id,
            model=args.model,
            rfp_url=args.rfp_url,
            ea_standard_eval_url=args.ea_standard_url,
            output_tokens=args.output_tokens,
            industry_standards=args.industry_standards,
            ministry_compliances=args.ministry_compliances,
            output_language=args.output_language
        )
        
        print("-" * 80)
        if result:
            print("✅ RFP Completeness Check completed successfully!")
            print(f"Result: {result}")
        else:
            import traceback

            error = traceback.format_exc()

            print("Error:-" ,error)
            print("❌ RFP Completeness Check failed!")
            if error_msg_for_user:
                print(f"User Error: {error_msg_for_user}")
            if technical_error_msg:
                print(f"Technical Error: {technical_error_msg}")
        
        return 0 if result else 1
        
    except Exception as e:

        print(f"❌ Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 