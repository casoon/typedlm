//! Derive macros for `typedlm`. Use them through the `typedlm` crate.

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprPath, Lit, LitStr, Meta, Type, parse_macro_input,
};

/// Implements `typedlm::Signature` for an input struct.
///
/// ```ignore
/// /// Classify a customer support ticket
/// #[derive(TypedLm, Serialize)]
/// #[lm(output = TicketClassification)]
/// struct ClassifyTicket { text: String }
/// ```
///
/// The description comes from the type's doc comment; `#[lm(description = "...")]`
/// overrides it. `#[lm(name = "...")]` overrides the program name (default: type name).
/// `#[lm(validate = path::to_fn)]` adds domain validation: a
/// `fn(&Output) -> Result<(), Vec<String>>`.
#[proc_macro_derive(TypedLm, attributes(lm))]
pub fn derive_typed_lm(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut output: Option<Type> = None;
    let mut name: Option<LitStr> = None;
    let mut description: Option<LitStr> = None;
    let mut validate: Option<ExprPath> = None;

    for attr in input.attrs.iter().filter(|a| a.path().is_ident("lm")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("output") {
                output = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("name") {
                name = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("description") {
                description = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("validate") {
                validate = Some(meta.value()?.parse()?);
            } else {
                return Err(meta.error("expected `output`, `name`, `description` or `validate`"));
            }
            Ok(())
        })?;
    }

    let ident = &input.ident;
    let output =
        output.ok_or_else(|| syn::Error::new_spanned(ident, "missing `#[lm(output = Type)]`"))?;
    let name = name.map_or_else(|| ident.to_string(), |n| n.value());
    let description = match description {
        Some(d) => d.value(),
        None => doc_comment(&input.attrs),
    };
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let validate = validate.map(|path| {
        quote! {
            fn validate(output: &Self::Output) -> ::core::result::Result<(), ::std::vec::Vec<::std::string::String>> {
                #path(output)
            }
        }
    });

    let from_text = single_string_field(input).map(|field| {
        quote! {
            impl ::core::convert::From<::std::string::String> for #ident {
                fn from(text: ::std::string::String) -> Self {
                    Self #field
                }
            }
            impl ::core::convert::From<&str> for #ident {
                fn from(text: &str) -> Self {
                    Self::from(::std::string::String::from(text))
                }
            }
        }
    });

    Ok(quote! {
        #from_text
        impl #impl_generics ::typedlm::Signature for #ident #ty_generics #where_clause {
            type Output = #output;
            const NAME: &'static str = #name;
            const DESCRIPTION: &'static str = #description;
            #validate
        }
    })
}

/// For a non-generic struct with exactly one `String` field, the constructor body
/// taking `text` — `{ field: text }` or `(text)`.
fn single_string_field(input: &DeriveInput) -> Option<proc_macro2::TokenStream> {
    let Data::Struct(data) = &input.data else {
        return None;
    };
    if !input.generics.params.is_empty() || data.fields.len() != 1 {
        return None;
    }
    let field = data.fields.iter().next()?;
    let Type::Path(path) = &field.ty else {
        return None;
    };
    if path.qself.is_some() || !path.path.is_ident("String") {
        return None;
    }
    Some(match &field.ident {
        Some(name) => quote! { { #name: text } },
        None => quote! { (text) },
    })
}

/// Joins `///` lines into one paragraph-preserving string.
fn doc_comment(attrs: &[Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(lit) => match &lit.lit {
                    Lit::Str(s) => Some(s.value().trim().to_string()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        })
        .collect();
    lines.join("\n").trim().to_string()
}
